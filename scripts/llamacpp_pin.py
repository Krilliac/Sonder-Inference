#!/usr/bin/env python3
"""llama.cpp single-engine pin and patch-series verifier (cmake/llamacpp-pin.json).

Subcommands
  hash    SHA-256 of a local file, or of a URL downloaded into a fresh empty
          temporary directory.
  verify  Check a source tarball against the manifest: the tarball SHA-256,
          every file's git blob SHA-1 against `git ls-tree -r <commit>` read
          from --git-dir, the SI patch series in third_party/ggml-patches/, and
          (when patches are present) clean `git apply --check -p1` against the
          verified tree.
  check   Fail when the manifest defines non-mainline tracks, when the patch
          directory / hashes disagree with the manifest, or when a literal
          llama.cpp tag, commit or SHA-256 in the docs disagrees with it.

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
PATCH_NAME_RE = re.compile(r"^(\d{4})-[a-z0-9][a-z0-9._-]*\.patch$")


class PinError(Exception):
    """A failed verification (exit 1)."""


class UsageError(Exception):
    """Bad input (exit 2)."""


def load_entry(manifest: Path, track: str | None) -> tuple[str, dict]:
    try:
        data = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise UsageError(f"cannot read manifest {manifest}: {exc}") from exc
    tracks = data.get("tracks")
    if not isinstance(tracks, dict) or list(tracks.keys()) != ["mainline"]:
        raise PinError(
            f"{manifest}: Sonder Inference allows only the single 'mainline' track "
            f"(separate forks are retired; got {list(tracks.keys()) if isinstance(tracks, dict) else type(tracks).__name__})"
        )
    if data.get("default_track", "mainline") != "mainline":
        raise PinError(f"{manifest}: default_track must be 'mainline'")
    effective_track = track or "mainline"
    if effective_track != "mainline":
        raise UsageError(
            f"track {effective_track!r} is retired: Sonder Inference builds only 'mainline' "
            f"plus third_party/ggml-patches/"
        )
    entry = tracks["mainline"]
    if not isinstance(entry, dict):
        raise UsageError(f"track 'mainline' not an object in {manifest}")
    for key, rx in (("commit", HEX40), ("sha256", HEX64)):
        if not isinstance(entry.get(key), str) or not rx.match(entry[key]):
            raise UsageError(f"track 'mainline': {key} missing or malformed")
    if not isinstance(entry.get("tag"), str) or not entry["tag"]:
        raise UsageError("track 'mainline': tag missing")
    return effective_track, entry


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


def _check_rel_path(raw: str, context: str) -> None:
    if "\\" in raw or raw.startswith("/") or re.match(r"^[A-Za-z]:", raw):
        raise PinError(f"unsafe path in {context}: {raw!r}")
    parts = raw.rstrip("/").split("/")
    if any(p in ("..", ".", "") for p in parts):
        raise PinError(f"unsafe path in {context}: {raw!r}")
    if parts[0] == ".git":
        raise PinError(f"forbidden .git path in {context}: {raw!r}")


def inspect_patch_bytes(data: bytes, patch_name: str) -> list[str]:
    if not data:
        raise PinError(f"patch {patch_name} is empty")
    if b"\x00" in data:
        raise PinError(f"binary patches are forbidden ({patch_name})")
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise PinError(f"patch {patch_name} is not valid UTF-8: {exc}") from exc
    if "GIT binary patch" in text:
        raise PinError(f"binary patches are forbidden ({patch_name})")
    if re.search(r"(?m)^(new file mode|old mode|deleted file mode) 120000\b", text):
        raise PinError(f"symlink mode 120000 is forbidden ({patch_name})")

    saw_minus = False
    saw_plus = False
    saw_hunk = False
    targets: list[str] = []
    seen_targets: set[str] = set()

    def _record(path_val: str) -> None:
        if path_val not in seen_targets:
            seen_targets.add(path_val)
            targets.append(path_val)

    for line in text.splitlines():
        if line.startswith("diff --git "):
            tokens = line.split()
            if len(tokens) != 4:
                raise PinError(f"malformed diff --git header in {patch_name}: {line!r}")
            for tok in tokens[2:]:
                clean = tok[2:] if tok.startswith(("a/", "b/")) else tok
                _check_rel_path(clean, patch_name)
        elif line.startswith("--- "):
            saw_minus = True
            target = line[4:].split("\t", 1)[0].strip()
            if target != "/dev/null":
                clean = target[2:] if target.startswith("a/") else target
                _check_rel_path(clean, patch_name)
                _record(clean)
        elif line.startswith("+++ "):
            saw_plus = True
            target = line[4:].split("\t", 1)[0].strip()
            if target != "/dev/null":
                clean = target[2:] if target.startswith("b/") else target
                _check_rel_path(clean, patch_name)
                _record(clean)
        elif line.startswith("@@ "):
            saw_hunk = True
        elif line.startswith(("rename from ", "rename to ", "copy from ", "copy to ")):
            target = line.split(" ", 2)[2].strip()
            _check_rel_path(target, patch_name)
            _record(target)
    if not (saw_minus and saw_plus):
        raise PinError(f"patch {patch_name} contains no unified diff headers")
    if not targets or not saw_hunk:
        raise PinError(f"patch {patch_name} touches no target files or contains no hunks")
    return targets


def validate_patch_series(entry: dict, root: Path) -> list[Path]:
    """Validate patch_dir and patches in manifest entry against disk; return ordered patch paths."""
    patch_dir_rel = entry.get("patch_dir", "third_party/ggml-patches")
    if not isinstance(patch_dir_rel, str) or not patch_dir_rel:
        raise PinError("manifest patch_dir must be a non-empty relative path")
    _check_rel_path(patch_dir_rel, "manifest patch_dir")
    patch_dir = root / patch_dir_rel
    if not patch_dir.is_dir():
        raise PinError(f"patch directory not found: {patch_dir}")

    patches = entry.get("patches", [])
    if not isinstance(patches, list):
        raise PinError("manifest patches must be a list")

    prev_seq = -1
    listed_names: set[str] = set()
    ordered_paths: list[Path] = []
    for idx, item in enumerate(patches):
        if not isinstance(item, dict):
            raise PinError(f"patches[{idx}] must be an object")
        fname = item.get("file")
        sha = item.get("sha256")
        summary = item.get("summary")
        upstream = item.get("upstream")
        if not isinstance(fname, str) or not PATCH_NAME_RE.match(fname):
            raise PinError(f"patches[{idx}].file invalid (expected NNNN-<slug>.patch): {fname!r}")
        m = PATCH_NAME_RE.match(fname)
        assert m is not None
        seq = int(m.group(1))
        if seq <= prev_seq:
            raise PinError(f"patch sequence numbers must strictly increase (at {fname!r})")
        prev_seq = seq
        if fname in listed_names:
            raise PinError(f"duplicate patch filename {fname!r}")
        if not isinstance(sha, str) or not HEX64.match(sha):
            raise PinError(f"patches[{idx}].sha256 malformed for {fname!r}")
        if not isinstance(summary, str) or not summary.strip():
            raise PinError(f"patches[{idx}].summary must be non-empty for {fname!r}")
        if not isinstance(upstream, str) or not upstream.strip():
            raise PinError(f"patches[{idx}].upstream must be non-empty for {fname!r}")

        pfile = patch_dir / fname
        if not pfile.is_file() or pfile.is_symlink():
            raise PinError(f"listed patch file missing or symlink: {pfile}")
        actual = sha256_file(pfile)
        if actual != sha:
            raise PinError(f"patch SHA-256 mismatch for {fname}: {actual} != {sha}")
        inspect_patch_bytes(pfile.read_bytes(), fname)
        listed_names.add(fname)
        ordered_paths.append(pfile)

    for disk_patch in sorted(patch_dir.glob("*.patch")):
        if disk_patch.name not in listed_names:
            raise PinError(f"unlisted patch file on disk: {disk_patch.name} in {patch_dir}")

    return ordered_paths


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


def tarball_blobs(tarball: Path, capture_regular: dict[str, bytes] | None = None) -> dict[str, str]:
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
                raw = f.read()
                blobs[rel] = git_blob_sha1(raw)
                if capture_regular is not None:
                    capture_regular[rel] = raw
            else:
                raise PinError(f"unsupported tar entry type for {member.name!r}")
    if len(tops) != 1:
        raise PinError("tarball must contain exactly one top-level directory")
    return blobs


def _file_state(path: Path) -> str:
    if path.is_file() and not path.is_symlink():
        return f"FILE:{sha256_file(path)}"
    return "MISSING"


def verify_patches_apply(
    regular_files: dict[str, bytes],
    patch_paths: list[Path],
    work_parent: Path | None = None,
) -> None:
    if not patch_paths:
        return
    with tempfile.TemporaryDirectory(
        prefix="llamacpp-patch-verify-",
        dir=str(work_parent) if work_parent is not None else None,
    ) as tmp_str:
        tmp_dir = Path(tmp_str).resolve()
        for rel, content in regular_files.items():
            _check_rel_path(rel, "verified tarball")
            dst = tmp_dir / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(content)
        env = os.environ.copy()
        env["GIT_CEILING_DIRECTORIES"] = str(tmp_dir.parent)
        for pfile in patch_paths:
            targets = inspect_patch_bytes(pfile.read_bytes(), pfile.name)
            before = {rel: _file_state(tmp_dir / rel) for rel in targets}
            for extra, expect_banner in ((["--check"], "Checking patch "), ([], "Applied patch ")):
                proc = subprocess.run(
                    [
                        "git",
                        "-c",
                        "core.autocrlf=false",
                        "-c",
                        "core.safecrlf=false",
                        "apply",
                        *extra,
                        "-v",
                        "-p1",
                        "--whitespace=nowarn",
                        str(pfile),
                    ],
                    cwd=tmp_dir,
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=60,
                )
                combined = f"{proc.stdout}\n{proc.stderr}"
                if proc.returncode != 0:
                    raise PinError(f"patch {pfile.name} failed to apply: {proc.stderr.strip()}")
                if "Skipped patch" in combined or expect_banner not in combined:
                    raise PinError(f"patch {pfile.name} skipped or matched 0 files: {combined.strip()}")
            for rel in targets:
                if _file_state(tmp_dir / rel) == before[rel]:
                    raise PinError(f"patch {pfile.name} left target {rel!r} unchanged")


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
    patch_paths = validate_patch_series(entry, Path(args.root))
    if args.commit != entry["commit"]:
        raise PinError(f"--commit {args.commit} does not match manifest commit {entry['commit']} (track {track})")
    tarball = Path(args.tarball)
    if not tarball.is_file():
        raise UsageError(f"tarball not found: {tarball}")
    actual = sha256_file(tarball)
    if actual != entry["sha256"]:
        raise PinError(f"tarball SHA-256 {actual} != manifest {entry['sha256']} (track {track})")
    want = git_tree(Path(args.git_dir), args.commit)
    captured: dict[str, bytes] | None = {} if patch_paths else None
    have = tarball_blobs(tarball, capture_regular=captured)
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
    if captured is not None:
        verify_patches_apply(captured, patch_paths)
    print(
        f"ok: {track} {entry['tag']} sha256, {len(want)} files, and {len(patch_paths)} patch(es) "
        f"match {args.commit}"
    )
    return 0


def cmd_check(args: argparse.Namespace) -> int:
    track, entry = load_entry(Path(args.manifest), args.track)
    root = Path(args.root)
    patch_paths = validate_patch_series(entry, root)
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
    print(f"ok: docs and {len(patch_paths)} patch(es) agree with manifest track {track} ({entry['tag']})")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    h = sub.add_parser("hash", help="SHA-256 of a local file or a downloaded URL")
    g = h.add_mutually_exclusive_group(required=True)
    g.add_argument("--file")
    g.add_argument("--url")
    h.set_defaults(func=cmd_hash)

    v = sub.add_parser("verify", help="verify a tarball and patch series against the manifest and a git object database")
    v.add_argument("--tarball", required=True)
    v.add_argument("--git-dir", required=True)
    v.add_argument("--commit", required=True)
    v.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    v.add_argument("--root", default=str(REPO_ROOT))
    v.add_argument("--track")
    v.set_defaults(func=cmd_verify)

    c = sub.add_parser("check", help="docs and patch directory must agree with the manifest")
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
