"""Deterministic prompt material: filler records, planted facts, tool schemas.

Filler follows kvb.py: "Record N: code NNNNN. <words>" lines from a fixed
vocabulary and a seeded RNG, so a run is reproducible and a planted fact is
the only place its answer appears.
"""

from __future__ import annotations

import json
import random
from typing import Callable

WORDS = ("system cache memory token context model layer tensor quant river mountain ledger harbor "
         "signal vector kernel thread buffer anchor lantern orbit meadow copper violet summit "
         "archive compass beacon cipher delta ember falcon glacier horizon island jasper").split()

VAULTS = ["Cobalt", "Juniper", "Marigold", "Tamarind", "Obsidian", "Saffron", "Verdigris", "Halcyon"]
PHRASE_A = ["amber", "silver", "crimson", "ivory", "indigo", "scarlet", "olive", "cyan"]
PHRASE_B = ["otter", "heron", "lynx", "badger", "falcon", "marten", "osprey", "ibex"]

DEFAULT_CHARS_PER_TOKEN = 4.0


class Sizer:
    """Measures text in tokens: the server's /tokenize when available, else an estimate."""

    def __init__(self, tokenize: Callable[[str], int | None] | None = None):
        self._tokenize = tokenize
        self.exact = False
        self.chars_per_token = DEFAULT_CHARS_PER_TOKEN
        if tokenize is not None:
            sample = records(random.Random(7), 40)
            n = tokenize(sample)
            if n:
                self.exact = True
                self.chars_per_token = len(sample) / n

    @property
    def method(self) -> str:
        return "tokenize" if self.exact else f"estimate({self.chars_per_token:.2f} chars/token)"

    def measure(self, text: str) -> int:
        if self.exact and self._tokenize is not None:
            n = self._tokenize(text)
            if n is not None:
                return n
        return int(len(text) / self.chars_per_token)

    def fit(self, parts: list[str], target_tokens: int, sep: str = "\n") -> list[str]:
        """Longest prefix of ``parts`` whose joined size is <= target_tokens."""
        est = [len(p) + len(sep) for p in parts]
        # Estimated upper bound first, then refine with measure() by bisection.
        lo, hi = 0, len(parts)
        total = 0
        for i, n in enumerate(est):
            total += n
            if total / self.chars_per_token > target_tokens * 1.25:
                hi = i
                break
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if self.measure(sep.join(parts[:mid])) <= target_tokens:
                lo = mid
            else:
                hi = mid - 1
        return parts[:lo]


def record_line(rnd: random.Random, i: int, words: int = 60) -> str:
    s = " ".join(rnd.choice(WORDS) for _ in range(words))
    return f"Record {i}: code {rnd.randint(10000, 99999)}. {s}."


def records(rnd: random.Random, n: int, start: int = 1) -> str:
    return "\n".join(record_line(rnd, start + i) for i in range(n))


def record_list(rnd: random.Random, approx_tokens: int, chars_per_token: float) -> list[str]:
    out = []
    chars = 0
    i = 0
    goal = approx_tokens * chars_per_token * 1.3 + 400
    while chars < goal:
        i += 1
        line = record_line(rnd, i)
        out.append(line)
        chars += len(line) + 1
    return out


def planted_facts(rnd: random.Random, n: int) -> list[dict]:
    names = VAULTS[:]
    rnd.shuffle(names)
    facts = []
    for k in range(n):
        vault = names[k % len(names)] + ("" if k < len(names) else f"-{k}")
        phrase = f"{rnd.choice(PHRASE_A)}-{rnd.choice(PHRASE_B)}-{rnd.randint(1000, 9999)}"
        facts.append({
            "vault": vault,
            "phrase": phrase,
            "text": f"IMPORTANT NOTE: the access phrase for vault {vault} is {phrase}.",
            "question": (f"What is the access phrase for vault {vault} mentioned in the notes above? "
                         "Reply with the phrase only."),
        })
    return facts


def plant(lines: list[str], facts: list[dict], depths: list[float]) -> tuple[list[str], list[float]]:
    """Insert each fact at its fractional depth (0 = start, 1 = end). Returns lines and actual depths."""
    out = list(lines)
    order = sorted(range(len(facts)), key=lambda k: depths[k], reverse=True)
    for k in order:  # deepest first so earlier indexes stay valid
        idx = min(len(out), max(0, round(len(lines) * depths[k])))
        out.insert(idx, facts[k]["text"])
    actual = []
    for f in facts:
        actual.append(round(out.index(f["text"]) / max(1, len(out) - 1), 3))
    return out, actual


TOOL_NAMES = ["read_file", "write_file", "edit_file", "list_dir", "grep", "glob", "run_tests",
              "run_build", "git_status", "git_diff", "git_log", "web_fetch", "search_symbols",
              "open_url", "format_code", "lint", "task_create", "task_update", "memory_search", "shell"]


def tool_schemas(n: int) -> list[dict]:
    out = []
    for i in range(n):
        name = TOOL_NAMES[i % len(TOOL_NAMES)] + ("" if i < len(TOOL_NAMES) else f"_{i}")
        out.append({
            "type": "function",
            "function": {
                "name": name,
                "description": (f"{name.replace('_', ' ').capitalize()} in the workspace. Paths are relative to "
                                "the repository root. Fails with a structured error when the target is missing, "
                                "outside the workspace, or larger than the configured limit."),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Workspace-relative path."},
                        "pattern": {"type": "string", "description": "Optional glob or regex filter."},
                        "limit": {"type": "integer", "minimum": 1, "maximum": 2000,
                                  "description": "Maximum number of results or lines."},
                        "dry_run": {"type": "boolean", "description": "Report the effect without applying it."},
                    },
                    "required": ["path"],
                },
            },
        })
    return out


def agent_system_prompt(sizer: Sizer, target_tokens: int, n_tools: int, seed: int) -> str:
    """Large, byte-stable system prompt: rules + tool schemas + handbook filler."""
    head = [
        "You are a careful software engineering agent working in a local repository.",
        "Follow the project rules. Prefer small, verifiable changes. Never invent tool results.",
        "When you need a tool, answer with a single line: CALL <tool_name> <json arguments>.",
        "Otherwise answer briefly in plain text.",
        "",
        "## Tools",
        json.dumps(tool_schemas(n_tools), indent=1, sort_keys=True),
        "",
        "## Project handbook",
    ]
    base = "\n".join(head)
    rest = max(0, target_tokens - sizer.measure(base))
    lines = record_list(random.Random(seed), rest, sizer.chars_per_token)
    lines = sizer.fit(lines, rest) if rest else []
    return base + "\n" + "\n".join(lines)


def tool_result_block(rnd: random.Random, turn: int, approx_tokens: int, chars_per_token: float) -> str:
    tool = TOOL_NAMES[turn % len(TOOL_NAMES)]
    lines = []
    chars = 0
    i = 0
    while chars < approx_tokens * chars_per_token:
        i += 1
        line = f"src/module_{turn}/part_{i}.py:{rnd.randint(1, 900)}: " + " ".join(rnd.choice(WORDS) for _ in range(14))
        lines.append(line)
        chars += len(line) + 1
    return f"Tool result ({tool}, turn {turn}):\n" + "\n".join(lines)
