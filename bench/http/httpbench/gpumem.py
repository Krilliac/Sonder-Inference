"""Per-process GPU memory via Windows PDH counters.

Method from the KV-fit bench (kvb.py): read
``\\GPU Process Memory(pid_<PID>_*)\\Dedicated Usage`` and ``Shared Usage``
for the server process and sum the nonzero adapter instances. Shared usage
growing well above the clean line means the driver spilled VRAM into system
memory, which shows up as a sudden decode slowdown rather than an error.

A clean load's shared usage is not flat: it rises about 1 MiB per 1024 ctx
(bench-kv-fit.md: Q3 148 MiB at 49k, IQ4 132 MiB at 32k). The spill flag
fires when shared usage exceeds the clean baseline by more than
``SPILL_MARGIN_MIB``.

Elsewhere (Linux, macOS, remote servers) sampling is skipped with a reason.
"""

from __future__ import annotations

import csv
import io
import os
import shutil
import subprocess
from typing import Any

SPILL_MARGIN_MIB = 256.0
MIB = float(2 ** 20)


def platform_supported() -> tuple[bool, str | None]:
    if os.name != "nt":
        return False, "PDH GPU counters are Windows-only"
    if not (shutil.which("powershell") or shutil.which("pwsh")):
        return False, "powershell not found"
    return True, None


def find_pids(process_name: str) -> list[int]:
    """PIDs of running processes with this image name (``.exe`` optional)."""
    name = process_name if process_name.lower().endswith(".exe") else process_name + ".exe"
    try:
        out = subprocess.check_output(
            ["tasklist", "/FO", "CSV", "/NH", "/FI", f"IMAGENAME eq {name}"],
            text=True, timeout=30, stderr=subprocess.DEVNULL)
    except (OSError, subprocess.SubprocessError):
        return []
    return parse_tasklist_csv(out, name)


def parse_tasklist_csv(out: str, name: str) -> list[int]:
    pids = []
    for row in csv.reader(io.StringIO(out)):
        if len(row) >= 2 and row[0].lower() == name.lower():
            try:
                pids.append(int(row[1]))
            except ValueError:
                pass
    return pids


def _ps() -> str:
    return shutil.which("powershell") or shutil.which("pwsh") or "powershell"


def counter_command(pids: list[int]) -> str:
    paths = []
    for pid in pids:
        paths.append(f"'\\GPU Process Memory(pid_{pid}_*)\\Shared Usage'")
        paths.append(f"'\\GPU Process Memory(pid_{pid}_*)\\Dedicated Usage'")
    return ("$c=(Get-Counter " + ",".join(paths) + " -ErrorAction SilentlyContinue).CounterSamples;"
            "$c | % { $_.InstanceName + '|' + ($_.Path -replace '.*\\\\','') + '|' + $_.CookedValue }")


def parse_counter_output(out: str) -> dict[str, Any]:
    """Sum 'instance|counter|bytes' lines into MiB; zero-valued adapters are ignored."""
    res: dict[str, Any] = {"shared_mib": 0.0, "dedicated_mib": 0.0, "raw": []}
    for line in out.strip().splitlines():
        parts = line.strip().split("|")
        if len(parts) != 3:
            continue
        inst, kind, val = parts
        try:
            mib = float(val) / MIB
        except ValueError:
            continue
        if mib <= 0:
            continue
        res["raw"].append(f"{inst}:{kind}={mib:.0f}")
        if "shared" in kind.lower():
            res["shared_mib"] += mib
        else:
            res["dedicated_mib"] += mib
    res["shared_mib"] = round(res["shared_mib"], 1)
    res["dedicated_mib"] = round(res["dedicated_mib"], 1)
    return res


def nvidia_smi_used_mib() -> float | None:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        out = subprocess.check_output([exe, "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                      text=True, timeout=30, stderr=subprocess.DEVNULL)
        return float(out.strip().splitlines()[0])
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


class GpuSampler:
    """Samples one server process (by PID or image name)."""

    def __init__(self, pid: int | None = None, process_name: str | None = None):
        self.pid = pid
        self.process_name = process_name
        ok, why = platform_supported()
        self.available = ok and (pid is not None or bool(process_name))
        self.reason = why if not ok else (None if self.available else "no --pid / --process-name given")
        self.samples: list[dict[str, Any]] = []

    def pids(self) -> list[int]:
        if self.pid is not None:
            return [self.pid]
        return find_pids(self.process_name) if self.process_name else []

    def sample(self, label: str) -> dict[str, Any] | None:
        if not self.available:
            return None
        pids = self.pids()
        rec: dict[str, Any] = {"label": label, "pids": pids}
        if not pids:
            rec["error"] = f"no process named {self.process_name!r}"
        else:
            try:
                out = subprocess.check_output([_ps(), "-NoProfile", "-NonInteractive", "-Command", counter_command(pids)],
                                              text=True, timeout=90, stderr=subprocess.DEVNULL)
                rec.update(parse_counter_output(out))
                if not rec["raw"]:
                    rec["error"] = "PDH returned no nonzero GPU Process Memory instances for " + ",".join(map(str, pids))
            except (OSError, subprocess.SubprocessError) as e:
                rec["error"] = f"{type(e).__name__}: {e}"
        rec["nvidia_smi_used_mib"] = nvidia_smi_used_mib()
        self.samples.append(rec)
        return rec


def expected_clean_shared(ref_mib: float, ref_ctx: int, ctx: int | None) -> float:
    """Clean shared-usage line: ref + 1 MiB per 1024 ctx above the reference ctx."""
    if not ctx:
        return ref_mib
    return ref_mib + max(0, ctx - ref_ctx) / 1024.0


def spill_check(sample: dict[str, Any] | None, baseline_mib: float | None,
                margin: float = SPILL_MARGIN_MIB) -> dict[str, Any] | None:
    """{'spill': bool, 'excess_mib': float} or None when either side is unknown."""
    if not sample or "shared_mib" not in sample or sample.get("error") or baseline_mib is None:
        return None
    excess = sample["shared_mib"] - baseline_mib
    return {"spill": excess > margin, "excess_mib": round(excess, 1), "margin_mib": margin}
