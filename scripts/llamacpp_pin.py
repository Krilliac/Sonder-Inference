#!/usr/bin/env python3
"""llama.cpp pin manifest tool (cmake/llamacpp-pin.json).

Subcommands
  hash    SHA-256 of a local file, or of a URL downloaded into a fresh empty
          temporary directory.
  verify  Check a source tarball against the manifest: the tarball SHA-256, then
          the git blob SHA-1 of every file against `git ls-tree -r <commit>`
          read from --git-dir (a worktree, a .git directory or a bare repository). Nothing from the tarball is written to disk or
          executed; entries are read in memory.
  check   Fail when a literal llama.cpp tag, commit or SHA-256 in the docs
          disagrees with the manifest.

Run as `python -I scripts/llamacpp_pin.py ...`. Exit 0 on success, 1 on a
verification/consistency failure, 2 on usage or input errors.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST = REPO_ROOT / "cmake" / "llamacpp-pin.json"
DOC_FILES = ("docs/LICENSE_REVIEW.md", "docs/integration/llamacpp.md")
HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
TAG_RE = re.compile(r"(?<![0-9A-Za-z.])b\d{4,6}(?![0-9A-Za-z])")
# Release tags (v1.0.0 onwards). Other dependencies also use vX.Y.Z tags, so these
# only count inside the llama.cpp scope (see cmd_check), unlike the bNNNNN build tags.
SEMVER_TAG_RE = re.compile(r"(?<![0-9A-Za-z.])v\d+\.\d+\.\d+(?![0-9A-Za-z]|\.\d)")


class PinError(Exception):
    """A failed verification (exit 1)."""


class UsageError(Exception):
    """Bad input (exit 2)."""


def load_entry(manifest: Path, track: str | None) -> tuple[str, dict]:
    try:
        data = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise UsageError(f"cannot read manifest {manifest}: {exc}") from exc
    track = track or data.get("default_track", "mainline")
    entry = data.get("tracks", {}).get(track)
    if not isinstance(entry, dict):
        raise UsageError(f"track {track!r} not in {manifest}")
    for key, rx in (("commit", HEX40), ("sha256", HEX64)):
        if not isinstance(entry.get(key), str) or not rx.match(entry[key]):
            raise UsageError(f"track {track!r}: {key} missing or malformed")
    if not isinstance(entry.get("tag"), str) or not entry["tag"]:
        raise UsageError(f"track {track!r}: tag missing")
    return track, entry


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_blob_sha1(data: bytes) -> str:
    h = hashlib.sha1(b"blob %d\0" % len(data))
    h.update(data)
    return h.hexdigest()


def git_tree(git_dir: Path, commit: str) -> dict[str, tuple[str, str]]:
    """path -> (mode, blob sha) for every blob/symlink in the commit."""
    try:
        out = subprocess.run(
            ["git", "-C", str(git_dir), "ls-tree", "-r", "-z", commit],
            check=True, capture_output=True, timeout=120,
        ).stdout
    except (OSError, subprocess.SubprocessError) as exc:
        detail = getattr(exc, "stderr", b"") or b""
        raise UsageError(
            f"git ls-tree failed for {commit} in {git_dir}: {exc} {detail.decode(errors='replace')}"
        ) from exc
    tree: dict[str, tuple[str, str]] = {}
    for rec in out.split(b"\0"):
        if not rec:
            continue
        meta, _, path = rec.partition(b"\t")
        mode, typ, sha = meta.decode().split(" ")
        if typ != "blob":  # submodule links are not in source tarballs
            continue
        tree[path.decode("utf-8", "surrogateescape")] = (mode, sha)
    return tree


def split_member(name: str) -> tuple[str, str]:
    """Return (top-level directory name, path below it; '' for the top entry).

    Rejects absolute, backslash, drive-letter and dot/empty-component paths.
    """
    if "\\" in name or name.startswith("/") or re.match(r"^[A-Za-z]:", name):
        raise PinError(f"unsafe path in tarball: {name!r}")
    parts = name.rstrip("/").split("/")
    if any(p in ("..", ".", "") for p in parts):
        raise PinError(f"unsafe path in tarball: {name!r}")
    return parts[0], "/".join(parts[1:])


def tarball_blobs(tarball: Path) -> dict[str, str]:
    """relative path -> git blob sha1, read in memory (no extraction to disk).

    The archive must have exactly one top-level directory shared by every
    member: CMake's extract step uses the extraction root as the source dir when
    there is more than one top-level entry, so anything else is rejected.
    """
    blobs: dict[str, str] = {}
    tops: set[str] = set()
    try:
        tf = tarfile.open(tarball, "r:*")
    except (OSError, tarfile.TarError) as exc:
        raise PinError(f"cannot open tarball {tarball}: {exc}") from exc
    with tf:
        for member in tf:
            top, rel = split_member(member.name)
            if not rel and not member.isdir():
                raise PinError(f"non-directory entry at the top level of the tarball: {member.name!r}")
            tops.add(top)
            if len(tops) > 1:
                raise PinError(f"tarball has more than one top-level entry: {sorted(tops)!r}")
            if member.isdir():
                continue
            if rel in blobs:
                raise PinError(f"duplicate entry in tarball: {rel!r}")
            if member.issym():
                blobs[rel] = git_blob_sha1(member.linkname.encode("utf-8"))
            elif member.isreg():
                f = tf.extractfile(member)
                if f is None:
                    raise PinError(f"unreadable entry: {member.name!r}")
                blobs[rel] = git_blob_sha1(f.read())
            else:
                raise PinError(f"unsupported tar entry type for {member.name!r}")
    if len(tops) != 1:
        raise PinError("tarball must contain exactly one top-level directory")
    return blobs


def cmd_hash(args: argparse.Namespace) -> int:
    if args.file:
        print(sha256_file(Path(args.file)))
        return 0
    if not args.url.startswith(("https://", "http://", "file://")):
        raise UsageError("--url must be an http(s) or file URL")
    with tempfile.TemporaryDirectory(prefix="llamacpp-pin-") as tmp:
        dst = Path(tmp) / "download.tar.gz"
        with urllib.request.urlopen(args.url, timeout=120) as resp, open(dst, "wb") as out:  # noqa: S310
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                out.write(chunk)
        print(sha256_file(dst))
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    track, entry = load_entry(Path(args.manifest), args.track)
    if args.commit != entry["commit"]:
        raise PinError(f"--commit {args.commit} does not match manifest commit {entry['commit']} (track {track})")
    tarball = Path(args.tarball)
    if not tarball.is_file():
        raise UsageError(f"tarball not found: {tarball}")
    actual = sha256_file(tarball)
    if actual != entry["sha256"]:
        raise PinError(f"tarball SHA-256 {actual} != manifest {entry['sha256']} (track {track})")
    want = git_tree(Path(args.git_dir), args.commit)
    have = tarball_blobs(tarball)
    problems: list[str] = []
    for path in sorted(want.keys() - have.keys()):
        problems.append(f"missing from tarball: {path}")
    for path in sorted(have.keys() - want.keys()):
        problems.append(f"extra in tarball: {path}")
    for path in sorted(want.keys() & have.keys()):
        if want[path][1] != have[path]:
            problems.append(f"content differs: {path}")
    if problems:
        shown = problems[:20]
        more = f" (+{len(problems) - len(shown)} more)" if len(problems) > len(shown) else ""
        raise PinError("tree mismatch:\n  " + "\n  ".join(shown) + more)
    print(f"ok: {track} {entry['tag']} sha256 and {len(want)} files match {args.commit}")
    return 0


def cmd_check(args: argparse.Namespace) -> int:
    track, entry = load_entry(Path(args.manifest), args.track)
    root = Path(args.root)
    problems: list[str] = []
    for rel in DOC_FILES:
        path = root / rel
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise UsageError(f"cannot read {path}: {exc}") from exc
        # LICENSE_REVIEW.md covers many dependencies: vX.Y.Z tag, commit and
        # SHA-256 literals only count inside the llama.cpp section (heading naming
        # llama.cpp up to the next heading). The integration doc is llama.cpp-only.
        # bNNNNN build tags are llama.cpp-specific and count everywhere.
        scoped = rel == "docs/LICENSE_REVIEW.md"
        in_section = not scoped
        for lineno, line in enumerate(text.splitlines(), 1):
            if scoped and line.lstrip().startswith("#"):
                in_section = "llama.cpp" in line
            for m in TAG_RE.finditer(line):
                if m.group(0) != entry["tag"]:
                    problems.append(f"{rel}:{lineno}: tag {m.group(0)} != manifest {entry['tag']}")
            if not in_section:
                continue
            for m in SEMVER_TAG_RE.finditer(line):
                if m.group(0) != entry["tag"]:
                    problems.append(f"{rel}:{lineno}: tag {m.group(0)} != manifest {entry['tag']}")
            for m in re.finditer(r"\b[0-9a-f]{40}\b", line):
                if m.group(0) != entry["commit"]:
                    problems.append(f"{rel}:{lineno}: commit {m.group(0)} != manifest {entry['commit']}")
            for m in re.finditer(r"\b[0-9a-f]{64}\b", line):
                if m.group(0) != entry["sha256"]:
                    problems.append(f"{rel}:{lineno}: sha256 {m.group(0)} != manifest {entry['sha256']}")
    if problems:
        raise PinError("docs disagree with manifest:\n  " + "\n  ".join(problems))
    print(f"ok: docs agree with manifest track {track} ({entry['tag']})")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    h = sub.add_parser("hash", help="SHA-256 of a local file or a downloaded URL")
    g = h.add_mutually_exclusive_group(required=True)
    g.add_argument("--file")
    g.add_argument("--url")
    h.set_defaults(func=cmd_hash)

    v = sub.add_parser("verify", help="verify a tarball against the manifest and a git object database")
    v.add_argument("--tarball", required=True)
    v.add_argument("--git-dir", required=True)
    v.add_argument("--commit", required=True)
    v.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    v.add_argument("--track")
    v.set_defaults(func=cmd_verify)

    c = sub.add_parser("check", help="docs literals must agree with the manifest")
    c.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    c.add_argument("--track")
    c.add_argument("--root", default=str(REPO_ROOT))
    c.set_defaults(func=cmd_check)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except PinError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    except UsageError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
