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


def _observation_values(source: dict) -> dict[str, float]:
    values = {}
    for name in ("cache_n", "prompt_n"):
        value = source.get(name)
        if isinstance(value, dict):
            value = value.get("mean")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            values[name] = value
    metrics = source.get("metrics") or {}
    accepted = source.get("accepted_per_position", metrics.get("accepted_per_position", {}))
    for position, value in (accepted or {}).items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            values[f"accepted position {position}"] = value
    child = source.get("child_log") or {}
    for name, value in child.get("counters", {}).items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            values[f"child log {name}"] = value
    return values


def _observation_extra(source: dict) -> list[str]:
    extra = []
    values = _observation_values(source)
    for name in ("cache_n", "prompt_n"):
        if name in values:
            extra.append(f"{name} {_f(values[name], digits=0)}")
    accepted = [f"{k.removeprefix('accepted position ')}:{_f(v, digits=0)}"
                for k, v in values.items() if k.startswith("accepted position ")]
    if accepted:
        extra.append("accepted/pos " + ",".join(accepted))
    child = [f"{k.removeprefix('child log ')}={_f(v, digits=0)}"
             for k, v in values.items() if k.startswith("child log ")]
    if child:
        extra.append("child log " + ",".join(child))
    for name in ("metrics", "child_log"):
        observation = source.get(name) or {}
        for field in ("errors", "resets", "unknown"):
            if observation.get(field):
                extra.append(f"{name} {field}: {observation[field]}")
        if observation.get("reset") or observation.get("partial"):
            extra.append(f"{name}: reset or partial observation")
    return extra


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
    prov = res.get("provenance")
    if isinstance(prov, dict):
        L.append("## Provenance")
        L.append("")
        L.append(f"- Source commit: `{prov.get('source_commit')}`")
        L.append(f"- Upstream revision/digest: `{prov.get('upstream_revision_or_digest')}`")
        L.append(f"- Model SHA-256: `{prov.get('model_sha256')}`; Draft SHA-256: `{prov.get('draft_sha256')}`")
        L.append(f"- Corpus revision: `{prov.get('corpus_revision')}`")
        sc = prov.get("sample_count") or {}
        L.append(f"- Samples: ok `{sc.get('ok')}/{sc.get('completed')}` (expected `{sc.get('expected')}`; "
                 f"cold `{sc.get('cold')}`, warm `{sc.get('warm')}`, unclassified `{sc.get('unclassified')}`)")
        eff = prov.get("effective_settings") or {}
        L.append(f"- Effective settings: models `{eff.get('observed_models')}`, "
                 f"speculative `{eff.get('server_speculative')}`, "
                 f"draft active `{eff.get('draft_observed_active')}`, "
                 f"prefix cache active `{eff.get('prefix_cache_observed_active')}`")
        vo = prov.get("validation_outcome") or {}
        L.append(f"- Validation outcome: `{vo.get('verdict')}` "
                 f"(pass `{vo.get('checks_passed')}`, warn `{vo.get('checks_warned')}`, "
                 f"fail `{vo.get('checks_failed')}`, skip `{vo.get('checks_skipped')}`; "
                 f"provenance_complete `{vo.get('provenance_complete')}`)")
        L.append("")
    L.append("## Checks")
    L.append("")
    L.append("| status | check | scenario | detail |")
    L.append("|---|---|---|---|")
    for c in res.get("checks") or []:
        st = c["status"].upper() if c["status"] in ("fail", "warn") else c["status"]
        L.append(f"| {st} | {c['check']} | {c.get('scenario') or ''} | {c['detail']} |")
    L.append("")
    cw_top = res.get("cold_vs_warm")
    if isinstance(cw_top, dict) and any((cw_top.get(k) or {}).get("requests") for k in ("cold", "warm", "unclassified")):
        L.append("## Cold vs Warm Summary")
        L.append("")
        L.append("Client-observed latency (TTFT / ITL) is reported separately from server backend compute (ms), "
                 "and committed output tokens are reported separately from proposed draft tokens.")
        L.append("")
        L.append("| cache state | ok/n | committed tok | proposed draft tok | client TTFT p50/p95 ms | "
                 "client ITL p50/p95 ms | backend compute p50/p95 ms | prefix hit |")
        L.append("|---|---|---|---|---|---|---|---|")
        for state in ("cold", "warm", "unclassified"):
            gs = cw_top.get(state) or {}
            if not gs.get("requests"):
                continue
            cli = gs.get("client_observed_ms") or {}
            be = gs.get("backend_compute_ms") or {}
            L.append("| " + " | ".join([
                state,
                f"{gs.get('ok')}/{gs.get('requests')}",
                _f(gs.get("committed_output_tokens_total"), digits=0),
                _f(gs.get("proposed_draft_tokens_total"), digits=0),
                f"{_f(_p(cli.get('ttft_ms'), 'p50'))} / {_f(_p(cli.get('ttft_ms'), 'p95'))}",
                f"{_f(_p(cli.get('itl_ms'), 'p50'), digits=2)} / {_f(_p(cli.get('itl_ms'), 'p95'), digits=2)}",
                f"{_f(_p(be.get('total_ms'), 'p50'))} / {_f(_p(be.get('total_ms'), 'p95'))}",
                _hit(gs),
            ]) + " |")
        L.append("")
    L.append("## Results")
    L.append("")
    agentx_scens = [s for s in (res.get("scenarios") or []) if isinstance(s.get("agentx_summary"), dict)]
    if agentx_scens:
        L.append("## AgentX Trace-Replay Summary")
        L.append("")
        L.append("Headline metric: **P90 interactivity tok/s** (`p90_interactivity_tps = 1000 / itl_ms.p90`, slow tail "
                 "pooled per concurrency level). Secondary per-request decode p50/p90 (`output_tps_per_user`, fast tail "
                 "at p90) is shown separately. Realized prefix hit is token-weighted `sum(cached_tokens) / sum(prompt_tokens)`.")
        L.append("")
        for s in agentx_scens:
            ax = s["agentx_summary"]
            scaled_list = ax.get("scaled_traces") or []
            clamped_n = ax.get("clamped_output_requests") or 0
            L.append(f"- `{s['name']}`: trace_fit=`{ax.get('trace_fit')}`, "
                     f"max_context_length={ax.get('max_context_length')} "
                     f"(effective {ax.get('effective_max_context_length')}), "
                     f"traces kept={ax.get('traces_kept')}/{ax.get('traces_loaded')}, "
                     f"scaled={len(scaled_list)}, skipped={ax.get('skipped_traces')}, "
                     f"out=0 clamped={clamped_n}")
            for st in scaled_list[:10]:
                L.append(f"  - scaled `{st['trace_id']}`: peak {st['original_peak_tokens']} -> "
                         f"{st['scaled_peak_tokens']} tokens (scale_factor={st['scale_factor']:.4f})")
        L.append("")
        L.append("| scenario | level | ok/n | sessions | P90 interactivity tok/s | P50 interactivity tok/s | "
                 "req decode p50/p90 (fast tail) | realized prefix hit | TTFT p50/p90 ms | ITL p50/p90 ms | "
                 "agg output tok/s | makespan (s) |")
        L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for s in agentx_scens:
            ax = s["agentx_summary"]
            for g, lv in (ax.get("levels") or {}).items():
                rh = "unknown" if lv.get("realized_prefix_hit_rate") is None else f"{lv['realized_prefix_hit_rate']:.3f}"
                L.append("| " + " | ".join([
                    s["name"],
                    g,
                    f"{lv.get('ok')}/{lv.get('requests')}",
                    _f(lv.get("sessions"), digits=0),
                    _f(lv.get("p90_interactivity_tps")),
                    _f(lv.get("p50_interactivity_tps")),
                    f"{_f(lv.get('req_decode_tps_p50'))} / {_f(lv.get('req_decode_tps_p90_fast_tail'))}",
                    rh,
                    f"{_f(lv.get('ttft_p50_ms'))} / {_f(lv.get('ttft_p90_ms'))}",
                    f"{_f(lv.get('itl_p50_ms'), digits=2)} / {_f(lv.get('itl_p90_ms'), digits=2)}",
                    _f(lv.get("aggregate_output_tps")),
                    _f(lv.get("makespan_s"), digits=2),
                ]) + " |")
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
            if gs.get("p90_interactivity_tps") is not None:
                extra.append(f"p90-interactivity {_f(gs['p90_interactivity_tps'])} tok/s")
            if gs.get("realized_prefix_hit_rate") is not None:
                extra.append(f"realized hit {gs['realized_prefix_hit_rate']:.3f}")
            if gs.get("proposed_draft_tokens_total") is not None:
                extra.append(f"proposed {_f(gs['proposed_draft_tokens_total'], digits=0)} draft tok")
            be = (gs.get("backend_compute_ms") or {}).get("total_ms")
            if _p(be, "p50") is not None:
                extra.append(f"backend {_f(_p(be, 'p50'))} ms")
            da = gs.get("draft_acceptance") or {}
            if da.get("n"):
                extra.append(f"draft acc {da['mean']:.2f}")
            extra.extend(_observation_extra(gs))
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
    observations = [("run", res)]
    for scenario in res.get("scenarios") or []:
        observations.append((scenario["name"], scenario))
        observations.extend((f"{scenario['name']}/{request.get('group', i)}", request)
                            for i, request in enumerate(scenario.get("requests") or []))
    observed = [(scope, _observation_extra(source)) for scope, source in observations]
    observed = [(scope, detail) for scope, detail in observed if detail]
    if observed:
        L.extend(["", "## Observations", "",
                  "Metrics are server-wide interval deltas; concurrent request windows overlap. "
                  "Child-log counters cover run/scenario intervals only.", ""])
        for scope, detail in observed:
            L.append(f"- `{scope}`: " + "; ".join(detail))
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
    ("TTFT p90 ms", ("ttft_s", "p90"), 1000, True),
    ("TTFT p95 ms", ("ttft_s", "p95"), 1000, True),
    ("ITL p50 ms", ("itl_s", "p50"), 1000, True),
    ("ITL p90 ms", ("itl_s", "p90"), 1000, True),
    ("ITL p95 ms", ("itl_s", "p95"), 1000, True),
    ("ITL p99 ms", ("itl_s", "p99"), 1000, True),
    ("e2e p50 s", ("e2e_s", "p50"), 1, True),
    ("e2e p90 s", ("e2e_s", "p90"), 1, True),
    ("backend compute p50 ms", ("backend_compute_ms", "total_ms", "p50"), 1, True),
    ("backend compute p95 ms", ("backend_compute_ms", "total_ms", "p95"), 1, True),
    ("backend prefill p50 ms", ("backend_compute_ms", "prefill_ms", "p50"), 1, True),
    ("backend decode p50 ms", ("backend_compute_ms", "decode_ms", "p50"), 1, True),
    ("committed output tokens", ("committed_output_tokens_total",), 1, False),
    ("proposed draft tokens", ("proposed_draft_tokens_total",), 1, False),
    ("prefill tok/s (client)", ("prefill_tps_client", "mean"), 1, False),
    ("prefill tok/s (server)", ("prefill_tps_server", "mean"), 1, False),
    ("decode tok/s (client)", ("decode_tps_client", "mean"), 1, False),
    ("decode tok/s (server)", ("decode_tps_server", "mean"), 1, False),
    ("P90 interactivity tok/s", ("p90_interactivity_tps",), 1, False),
    ("P50 interactivity tok/s", ("p50_interactivity_tps",), 1, False),
    ("per-user decode tok/s p50", ("output_tps_per_user", "p50"), 1, False),
    ("per-user decode tok/s p90 (fast tail)", ("output_tps_per_user", "p90"), 1, False),
    ("prefix hit", ("prefix_hit", "mean"), 1, False),
    ("realized prefix hit", ("realized_prefix_hit_rate",), 1, False),
    ("recall", ("recall",), 1, False),
    ("aggregate tok/s", ("aggregate_decode_tps",), 1, False),
    ("makespan s", ("makespan_s",), 1, True),
]


def _compare_observations(a: dict, b: dict, scenario: str, group: str) -> list[dict]:
    av, bv = _observation_values(a), _observation_values(b)
    rows = []
    for name in sorted(set(av) | set(bv)):
        va, vb = av.get(name), bv.get(name)
        rows.append({"scenario": scenario, "group": group, "metric": name, "a": va, "b": vb,
                     "delta": vb - va if va is not None and vb is not None else None,
                     "delta_pct": (vb - va) / va * 100 if va not in (None, 0) and vb is not None else None,
                     "better": None})
    return rows


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
        rows.extend(_compare_observations(ga[key], gb[key], key[0], key[1]))
    # Never sum overlapping request windows into scenario or run counters.
    sa = {s["name"]: s for s in a.get("scenarios") or []}
    sb = {s["name"]: s for s in b.get("scenarios") or []}
    for name in sorted(set(sa) & set(sb)):
        rows.extend(_compare_observations(sa[name], sb[name], name, "scenario"))
        cwa = sa[name].get("cold_vs_warm") or {}
        cwb = sb[name].get("cold_vs_warm") or {}
        for state in ("cold", "warm"):
            gsa, gsb = cwa.get(state) or {}, cwb.get(state) or {}
            if not (gsa.get("ok") or gsb.get("ok")):
                continue
            for label, path, scale, lower in COMPARE_METRICS:
                va, vb = _get(gsa, path), _get(gsb, path)
                if va is None and vb is None:
                    continue
                row = {"scenario": name, "group": f"({state})", "metric": label,
                       "a": va * scale if va is not None else None, "b": vb * scale if vb is not None else None,
                       "delta": None, "delta_pct": None, "better": None}
                if va is not None and vb is not None:
                    row["delta"] = (vb - va) * scale
                    row["delta_pct"] = (vb - va) / va * 100 if va else None
                    if vb != va:
                        row["better"] = (vb < va) if lower else (vb > va)
                rows.append(row)
    rows.extend(_compare_observations(a, b, "(run)", "observations"))
    L = [f"# A/B: `{a.get('label') or 'A'}` vs `{b.get('label') or 'B'}`", ""]
    L.append(f"- A: `{a['target']['base_url']}` model `{a['target'].get('model')}` build "
             f"`{(a.get('server') or {}).get('build_info')}` verdict {a.get('verdict')}")
    L.append(f"- B: `{b['target']['base_url']}` model `{b['target'].get('model')}` build "
             f"`{(b.get('server') or {}).get('build_info')}` verdict {b.get('verdict')}")
    pa, pb = a.get("provenance") or {}, b.get("provenance") or {}
    if pa or pb:
        L.append(f"- Provenance A: commit `{pa.get('source_commit')}`, model `{pa.get('model_sha256')}`, "
                 f"draft `{pa.get('draft_sha256')}`, corpus `{pa.get('corpus_revision')}`")
        L.append(f"- Provenance B: commit `{pb.get('source_commit')}`, model `{pb.get('model_sha256')}`, "
                 f"draft `{pb.get('draft_sha256')}`, corpus `{pb.get('corpus_revision')}`")
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
