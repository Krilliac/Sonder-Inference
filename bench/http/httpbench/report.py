"""Markdown summary and A/B comparison of result files."""

from __future__ import annotations

from typing import Any


def _f(x: Any, scale: float = 1.0, digits: int = 1) -> str:
    if x is None:
        return "–"
    if isinstance(x, bool):
        return "yes" if x else "no"
    if isinstance(x, (int, float)):
        return f"{x * scale:.{digits}f}"
    return str(x)


def _p(s: dict | None, key: str) -> Any:
    if not s or not s.get("n"):
        return None
    return s.get(key)


def _hit(g: dict) -> str:
    h = g.get("prefix_hit")
    if h is None:
        return "unknown"
    return f"{h['mean']:.2f}"


def render_markdown(res: dict) -> str:
    L: list[str] = []
    srv = res.get("server") or {}
    L.append(f"# HTTP bench: {res.get('label') or 'run'}")
    L.append("")
    L.append(f"- Target: `{res['target']['base_url']}`, model `{res['target'].get('model')}`")
    L.append(f"- Server build: `{srv.get('build_info')}`; n_ctx {srv.get('n_ctx')} ({srv.get('ctx_source')}); "
             f"models {srv.get('model_ids')}")
    L.append(f"- Started {res.get('started_at')}, wall {_f(res.get('wall_s'))} s; prompt sizing {res.get('prompt_sizing')}")
    gpu = res.get("gpu") or {}
    if gpu.get("available"):
        L.append(f"- GPU sampling: pid {gpu.get('pid')} / process {gpu.get('process_name')!r}; shared baseline "
                 f"{gpu.get('shared_baseline_mib')} MiB ({gpu.get('baseline_source')})")
    else:
        L.append(f"- GPU sampling: off ({gpu.get('reason')})")
    L.append(f"- **Verdict: {res.get('verdict', '?').upper()}**")
    L.append("")
    L.append("## Checks")
    L.append("")
    L.append("| status | check | scenario | detail |")
    L.append("|---|---|---|---|")
    for c in res.get("checks") or []:
        st = c["status"].upper() if c["status"] in ("fail", "warn") else c["status"]
        L.append(f"| {st} | {c['check']} | {c.get('scenario') or ''} | {c['detail']} |")
    L.append("")
    L.append("## Results")
    L.append("")
    L.append("Times in ms unless marked. ITL is per token (chunk gap / tokens in chunk). "
             "Prefix hit is cached / prompt tokens; `unknown` means the server did not report cached tokens.")
    L.append("")
    L.append("| scenario | group | ok/n | prompt tok | gen tok | TTFT p50 | TTFT p95 | ITL p50 | ITL p95 | ITL p99 | "
             "e2e p50 (s) | prefill tok/s client / server | decode tok/s client / server | prefix hit | recall | extra |")
    L.append("|" + "---|" * 16)
    for s in res.get("scenarios") or []:
        for g, gs in (s.get("groups") or {}).items():
            extra = []
            if "aggregate_decode_tps" in gs:
                extra.append(f"agg {_f(gs['aggregate_decode_tps'])} tok/s, makespan {_f(gs.get('makespan_s'), digits=2)} s")
            da = gs.get("draft_acceptance") or {}
            if da.get("n"):
                extra.append(f"draft acc {da['mean']:.2f}")
            rec = "–" if gs.get("recall") is None else f"{gs['recall'] * gs['recall_n']:.0f}/{gs['recall_n']}"
            L.append("| " + " | ".join([
                s["name"], g, f"{gs.get('ok')}/{gs.get('requests')}",
                _f(_p(gs.get("prompt_tokens"), "mean"), digits=0), _f(gs.get("completion_tokens_total"), digits=0),
                _f(_p(gs.get("ttft_s"), "p50"), 1000), _f(_p(gs.get("ttft_s"), "p95"), 1000),
                _f(_p(gs.get("itl_s"), "p50"), 1000, 2), _f(_p(gs.get("itl_s"), "p95"), 1000, 2),
                _f(_p(gs.get("itl_s"), "p99"), 1000, 2), _f(_p(gs.get("e2e_s"), "p50"), digits=2),
                f"{_f(_p(gs.get('prefill_tps_client'), 'mean'))} / {_f(_p(gs.get('prefill_tps_server'), 'mean'))}",
                f"{_f(_p(gs.get('decode_tps_client'), 'mean'))} / {_f(_p(gs.get('decode_tps_server'), 'mean'))}",
                _hit(gs), rec, "; ".join(extra)]) + " |")
        for sk in s.get("skipped") or []:
            L.append(f"| {s['name']} | {sk['size']} | skipped | | | | | | | | | | | | | {sk['reason']} |")
    samples = gpu.get("samples") or []
    if samples:
        L.append("")
        L.append("## GPU memory (PDH, per process)")
        L.append("")
        L.append("| sample | dedicated MiB | shared MiB | nvidia-smi used MiB | note |")
        L.append("|---|---|---|---|---|")
        for x in samples:
            L.append(f"| {x['label']} | {_f(x.get('dedicated_mib'), digits=0)} | {_f(x.get('shared_mib'), digits=0)} | "
                     f"{_f(x.get('nvidia_smi_used_mib'), digits=0)} | {x.get('error') or ''} |")
    L.append("")
    return "\n".join(L)


# ---------------------------------------------------------------------- compare

# (label, path into a group summary, scale, lower_is_better)
COMPARE_METRICS = [
    ("TTFT p50 ms", ("ttft_s", "p50"), 1000, True),
    ("TTFT p95 ms", ("ttft_s", "p95"), 1000, True),
    ("ITL p50 ms", ("itl_s", "p50"), 1000, True),
    ("ITL p95 ms", ("itl_s", "p95"), 1000, True),
    ("ITL p99 ms", ("itl_s", "p99"), 1000, True),
    ("e2e p50 s", ("e2e_s", "p50"), 1, True),
    ("prefill tok/s (client)", ("prefill_tps_client", "mean"), 1, False),
    ("prefill tok/s (server)", ("prefill_tps_server", "mean"), 1, False),
    ("decode tok/s (client)", ("decode_tps_client", "mean"), 1, False),
    ("decode tok/s (server)", ("decode_tps_server", "mean"), 1, False),
    ("prefix hit", ("prefix_hit", "mean"), 1, False),
    ("recall", ("recall",), 1, False),
    ("aggregate tok/s", ("aggregate_decode_tps",), 1, False),
    ("makespan s", ("makespan_s",), 1, True),
]


def _get(gs: dict, path: tuple) -> float | None:
    cur: Any = gs
    for k in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(k)
    if isinstance(cur, bool) or not isinstance(cur, (int, float)):
        return None
    return float(cur)


def compare(a: dict, b: dict) -> tuple[str, list[dict]]:
    """Markdown diff of B against A, plus the rows as data."""
    rows: list[dict] = []
    ga = {(s["name"], g): gs for s in a.get("scenarios") or [] for g, gs in (s.get("groups") or {}).items()}
    gb = {(s["name"], g): gs for s in b.get("scenarios") or [] for g, gs in (s.get("groups") or {}).items()}
    for key in [k for k in ga if k in gb]:
        for label, path, scale, lower in COMPARE_METRICS:
            va, vb = _get(ga[key], path), _get(gb[key], path)
            if va is None and vb is None:
                continue
            row = {"scenario": key[0], "group": key[1], "metric": label,
                   "a": va * scale if va is not None else None, "b": vb * scale if vb is not None else None,
                   "delta": None, "delta_pct": None, "better": None}
            if va is not None and vb is not None:
                row["delta"] = (vb - va) * scale
                row["delta_pct"] = (vb - va) / va * 100 if va else None
                if vb != va:
                    row["better"] = (vb < va) if lower else (vb > va)
            rows.append(row)
    L = [f"# A/B: `{a.get('label') or 'A'}` vs `{b.get('label') or 'B'}`", ""]
    L.append(f"- A: `{a['target']['base_url']}` model `{a['target'].get('model')}` build "
             f"`{(a.get('server') or {}).get('build_info')}` verdict {a.get('verdict')}")
    L.append(f"- B: `{b['target']['base_url']}` model `{b['target'].get('model')}` build "
             f"`{(b.get('server') or {}).get('build_info')}` verdict {b.get('verdict')}")
    only_a = sorted({k for k in ga if k not in gb})
    only_b = sorted({k for k in gb if k not in ga})
    if only_a or only_b:
        L.append(f"- Groups only in A: {only_a or 'none'}; only in B: {only_b or 'none'}")
    L.append("")
    L.append("| scenario | group | metric | A | B | delta | delta % | |")
    L.append("|---|---|---|---|---|---|---|---|")
    for r in rows:
        mark = "" if r["better"] is None else ("better" if r["better"] else "worse")
        L.append(f"| {r['scenario']} | {r['group']} | {r['metric']} | {_f(r['a'], digits=3)} | {_f(r['b'], digits=3)} | "
                 f"{_f(r['delta'], digits=3)} | {_f(r['delta_pct'])} | {mark} |")
    L.append("")
    return "\n".join(L), rows
