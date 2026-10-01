#!/usr/bin/env python3
"""HTTP benchmark harness for OpenAI-compatible inference servers.

    python bench/http/bench_http.py run --base-url http://127.0.0.1:8080 --label raw-llama
    python bench/http/bench_http.py compare A.json B.json
    python bench/http/bench_http.py scenarios          # print the built-in scenario config

Standard library only. See bench/http/README.md.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from httpbench import __version__  # noqa: E402
from httpbench.client import Target, parse_header  # noqa: E402
from httpbench.gpumem import GpuSampler, expected_clean_shared  # noqa: E402
from httpbench.report import compare, render_markdown  # noqa: E402
from httpbench.runner import BUILTIN_SCENARIOS, Runner, default_out_prefix, load_config  # noqa: E402


def _csv_ints(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def _header(s: str) -> tuple[str, str]:
    try:
        return parse_header(s)
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"invalid header: {e}") from e


def _redact_argv(argv: list[str]) -> list[str]:
    out = []
    redact_next = False
    for arg in argv:
        option = arg.split("=", 1)[0]
        sensitive = option.startswith("--") and any(flag.startswith(option) for flag in ("--header", "--api-key"))
        if redact_next:
            out.append("<redacted>")
            redact_next = False
        elif sensitive:
            out.append(option + "=<redacted>" if "=" in arg else arg)
            redact_next = "=" not in arg
        else:
            out.append(arg)
    return out


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="bench_http.py", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", action="version", version=f"bench_http {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="run scenarios against a server")
    r.add_argument("--base-url", required=True,
                   help="server root or OpenAI prefix, e.g. http://127.0.0.1:8080 or http://127.0.0.1:11437/v1")
    r.add_argument("--model", help="model id (default: first id from /v1/models)")
    r.add_argument("--label", default="", help="run label, used in file names and reports")
    r.add_argument("--scenarios", help="scenario config JSON (default: built-in set; see `scenarios`)")
    r.add_argument("--only", help="comma-separated scenario names to run")
    r.add_argument("--api", choices=["chat", "completion"],
                   help="force the API for every scenario (completion = llama-server /completion, raw prompt)")
    r.add_argument("--long-sizes", type=_csv_ints, help="override long_context sizes, e.g. 8192,16384")
    r.add_argument("--max-tokens", type=int, help="override max_tokens for every scenario")
    r.add_argument("--ctx-size", type=int, help="server context size (default: from /props or Sonder identity)")
    r.add_argument("--no-think", action="store_true",
                   help='send chat_template_kwargs {"enable_thinking": false} (Qwen3-style templates)')
    r.add_argument("--extra-body", help="JSON object merged into every request body")
    r.add_argument("--api-key", default=os.environ.get("BENCH_HTTP_API_KEY"),
                   help="bearer token (default: env BENCH_HTTP_API_KEY)")
    r.add_argument("--token-file", help="read the bearer token from this file (sonder-infer serve --token-file)")
    r.add_argument("--header", type=_header, action="append", default=[], metavar="K=V",
                   help="request header (repeatable; scenario headers override case-insensitively)")
    r.add_argument("--metrics-url", help="full child /metrics URL, scraped around each request (no auth forwarded)")
    r.add_argument("--child-log", metavar="PATH", help="count new child log events during the run")
    r.add_argument("--timeout", type=float, default=900.0, help="per-request socket timeout, seconds")
    g = r.add_argument_group("GPU memory (Windows PDH)")
    g.add_argument("--pid", type=int, help="server process id to sample")
    g.add_argument("--process-name", help="server image name to sample, e.g. llama-server")
    g.add_argument("--shared-baseline-mib", type=float,
                   help="clean shared-usage baseline in MiB (default: first sample of the run)")
    g.add_argument("--clean-line", metavar="MIB@CTX",
                   help="derive the baseline from a measured clean point, +1 MiB per 1024 ctx, e.g. 148@49152")
    r.add_argument("--out", help="output path prefix (writes PREFIX.json and PREFIX.md); default bench/http/out/<stamp>-<label>")
    r.add_argument("--strict", action="store_true", help="exit 1 on warnings too")
    r.add_argument("--quiet", action="store_true", help="no progress lines on stderr")

    c = sub.add_parser("compare", help="diff two result files (B against A)")
    c.add_argument("a")
    c.add_argument("b")
    c.add_argument("--out", help="also write the markdown diff here")
    c.add_argument("--json", action="store_true", help="print rows as JSON instead of markdown")

    sub.add_parser("scenarios", help="print the built-in scenario config as JSON")
    return ap


def cmd_run(args: argparse.Namespace) -> int:
    try:
        cfg = load_config(args.scenarios)
    except (OSError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    cfg = copy.deepcopy(cfg)
    if args.only:
        want = {x.strip() for x in args.only.split(",") if x.strip()}
        cfg["scenarios"] = [s for s in cfg["scenarios"] if (s.get("name") or s["kind"]) in want]
        missing = want - {s.get("name") or s["kind"] for s in cfg["scenarios"]}
        if missing:
            print(f"error: unknown scenario(s) {sorted(missing)}", file=sys.stderr)
            return 2
    for s in cfg["scenarios"]:
        if args.api:
            s["api"] = args.api
        if args.max_tokens:
            s["max_tokens"] = args.max_tokens
            s["_max_tokens_explicit"] = True
        if args.long_sizes and s["kind"] == "long_context":
            s["sizes"] = args.long_sizes
    extra: dict = {}
    if args.extra_body:
        try:
            extra = json.loads(args.extra_body)
            assert isinstance(extra, dict)
        except (ValueError, AssertionError):
            print("error: --extra-body must be a JSON object", file=sys.stderr)
            return 2
    if args.no_think:
        extra.setdefault("chat_template_kwargs", {})["enable_thinking"] = False
    key = args.api_key
    if args.token_file:
        with open(args.token_file, encoding="utf-8") as f:
            key = f.read().strip()
    try:
        headers = {}
        for name, value in args.header:
            headers = {k: v for k, v in headers.items() if k.lower() != name.lower()}
            headers[name] = value
        target = Target(args.base_url, api_key=key, timeout=args.timeout, headers=headers)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    sampler = GpuSampler(pid=args.pid, process_name=args.process_name)
    baseline = args.shared_baseline_mib
    try:
        runner = Runner(target, model=args.model, extra_body=extra, sampler=sampler, ctx_size=args.ctx_size,
                        shared_baseline_mib=baseline, log=(lambda m: None) if args.quiet else None,
                        metrics_url=args.metrics_url, child_log=args.child_log)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    if args.clean_line:
        try:
            mib, ctx = args.clean_line.split("@")
            ref_mib, ref_ctx = float(mib), int(ctx)
        except ValueError:
            print("error: --clean-line must look like 148@49152", file=sys.stderr)
            return 2
        runner.identify()  # learn n_ctx before the run to place the line
        runner.shared_baseline_mib = expected_clean_shared(ref_mib, ref_ctx, runner.ctx_size)
        runner.baseline_source = f"clean line {args.clean_line} at n_ctx {runner.ctx_size}"
    results = runner.run(cfg, label=args.label, argv=_redact_argv(sys.argv[1:]))
    prefix = args.out or default_out_prefix(args.label)
    os.makedirs(os.path.dirname(os.path.abspath(prefix)), exist_ok=True)
    with open(prefix + ".json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=1, default=str)
    md = render_markdown(results)
    with open(prefix + ".md", "w", encoding="utf-8") as f:
        f.write(md)
    print(md)
    print(f"wrote {prefix}.json and {prefix}.md", file=sys.stderr)
    if results["verdict"] == "fail" or (args.strict and results["verdict"] == "warn"):
        return 1
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    try:
        with open(args.a, encoding="utf-8") as f:
            a = json.load(f)
        with open(args.b, encoding="utf-8") as f:
            b = json.load(f)
    except (OSError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    md, rows = compare(a, b)
    if args.json:
        print(json.dumps(rows, indent=1))
    else:
        print(md)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(md)
    if not rows:
        print("warning: no common scenario/group metrics to compare", file=sys.stderr)
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.cmd == "run":
        return cmd_run(args)
    if args.cmd == "compare":
        return cmd_compare(args)
    print(json.dumps(BUILTIN_SCENARIOS, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
