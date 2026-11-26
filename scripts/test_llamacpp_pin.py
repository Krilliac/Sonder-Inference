"""Tests for scripts/llamacpp_pin.py (run: python -I -m unittest scripts.test_llamacpp_pin)."""
from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import llamacpp_pin as pin  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
GIT_ENV = {
    "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.invalid",
    "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example.invalid",
}

FILES = {
    "README.md": b"hello\n",
    "src/a.c": b"int a(void) { return 1; }\n",
    "src/sub/b.c": b"int b(void) { return 2; }\n",
}


def git(cwd: Path, *args: str) -> str:
    env = dict(os.environ, **GIT_ENV)
    return subprocess.run(["git", *args], cwd=cwd, env=env, check=True,
                          capture_output=True, text=True).stdout.strip()


def make_tarball(path: Path, files: dict[str, bytes], top: str = "llama.cpp-x") -> None:
    with tarfile.open(path, "w:gz") as tf:
        for name, data in files.items():
            ti = tarfile.TarInfo(f"{top}/{name}")
            ti.size = len(data)
            tf.addfile(ti, io.BytesIO(data))


def write_manifest(path: Path, commit: str, sha256: str, tag: str = "bTEST") -> None:
    path.write_text(json.dumps({
        "schema": 1, "default_track": "mainline",
        "tracks": {"mainline": {"tag": tag, "commit": commit, "sha256": sha256,
                                "url": "https://example.invalid/x.tar.gz"}},
    }), encoding="utf-8")


class VerifyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if shutil.which("git") is None:
            raise unittest.SkipTest("git not available")

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.repo = self.tmp / "repo"
        self.repo.mkdir()
        git(self.repo, "init", "-q")
        for name, data in FILES.items():
            p = self.repo / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(data)
        git(self.repo, "add", "-A")
        git(self.repo, "-c", "core.autocrlf=false", "commit", "-q", "-m", "x")
        self.commit = git(self.repo, "rev-parse", "HEAD")
        self.git_dir = self.repo / ".git"
        self.tarball = self.tmp / "src.tar.gz"
        make_tarball(self.tarball, FILES)
        self.manifest = self.tmp / "pin.json"
        self._write_manifest()

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _write_manifest(self) -> None:
        write_manifest(self.manifest, self.commit, pin.sha256_file(self.tarball))

    def run_verify(self) -> int:
        err = io.StringIO()
        old = sys.stderr, sys.stdout
        sys.stderr, sys.stdout = err, io.StringIO()
        try:
            return pin.main(["verify", "--tarball", str(self.tarball), "--git-dir", str(self.git_dir),
                             "--commit", self.commit, "--manifest", str(self.manifest)])
        finally:
            sys.stderr, sys.stdout = old

    def test_matching_tarball_passes(self) -> None:
        self.assertEqual(self.run_verify(), 0)

    def test_flipped_byte_fails(self) -> None:
        files = dict(FILES)
        data = bytearray(files["src/a.c"])
        data[0] ^= 0x01
        files["src/a.c"] = bytes(data)
        make_tarball(self.tarball, files)
        self._write_manifest()  # SHA-256 matches the tampered tarball: tree check must catch it
        self.assertEqual(self.run_verify(), 1)

    def test_wrong_manifest_sha_fails(self) -> None:
        write_manifest(self.manifest, self.commit, "0" * 64)
        self.assertEqual(self.run_verify(), 1)

    def test_missing_file_fails(self) -> None:
        files = dict(FILES)
        del files["src/sub/b.c"]
        make_tarball(self.tarball, files)
        self._write_manifest()
        self.assertEqual(self.run_verify(), 1)

    def test_extra_file_fails(self) -> None:
        files = dict(FILES)
        files["evil.sh"] = b"echo hi\n"
        make_tarball(self.tarball, files)
        self._write_manifest()
        self.assertEqual(self.run_verify(), 1)

    def test_path_traversal_fails(self) -> None:
        files = dict(FILES)
        files["../escape"] = b"x"
        make_tarball(self.tarball, files)
        self._write_manifest()
        self.assertEqual(self.run_verify(), 1)

    def test_commit_must_match_manifest(self) -> None:
        write_manifest(self.manifest, "1" * 40, pin.sha256_file(self.tarball))
        self.assertEqual(self.run_verify(), 1)

    def test_unknown_commit_is_usage_error(self) -> None:
        bad = "2" * 40
        write_manifest(self.manifest, bad, pin.sha256_file(self.tarball))
        self.commit = bad
        self.assertEqual(self.run_verify(), 2)


class HashTests(unittest.TestCase):
    def test_hash_file(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "f.bin"
            f.write_bytes(b"abc")
            out = io.StringIO()
            old, sys.stdout = sys.stdout, out
            try:
                rc = pin.main(["hash", "--file", str(f)])
            finally:
                sys.stdout = old
            self.assertEqual(rc, 0)
            self.assertEqual(out.getvalue().strip(), hashlib.sha256(b"abc").hexdigest())

    def test_hash_url_downloads_into_fresh_dir(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "f.bin"
            f.write_bytes(b"abc")
            out = io.StringIO()
            old, sys.stdout = sys.stdout, out
            try:
                rc = pin.main(["hash", "--url", f.as_uri()])
            finally:
                sys.stdout = old
            self.assertEqual(rc, 0)
            self.assertEqual(out.getvalue().strip(), hashlib.sha256(b"abc").hexdigest())


class CheckTests(unittest.TestCase):
    def run_check(self, *args: str) -> int:
        old = sys.stderr, sys.stdout
        sys.stderr, sys.stdout = io.StringIO(), io.StringIO()
        try:
            return pin.main(["check", *args])
        finally:
            sys.stderr, sys.stdout = old

    def test_repo_docs_agree_with_manifest(self) -> None:
        self.assertEqual(self.run_check(), 0)

    def test_manifest_has_mainline_track(self) -> None:
        track, entry = pin.load_entry(pin.DEFAULT_MANIFEST, None)
        self.assertEqual(track, "mainline")
        self.assertRegex(entry["tag"], r"^b\d+$")

    def test_changed_manifest_tag_is_detected(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            m = Path(d) / "pin.json"
            _, entry = pin.load_entry(pin.DEFAULT_MANIFEST, None)
            write_manifest(m, entry["commit"], entry["sha256"], tag="b99999")
            self.assertEqual(self.run_check("--manifest", str(m)), 1)

    def test_changed_manifest_sha_is_detected(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            m = Path(d) / "pin.json"
            _, entry = pin.load_entry(pin.DEFAULT_MANIFEST, None)
            write_manifest(m, entry["commit"], "f" * 64, tag=entry["tag"])
            self.assertEqual(self.run_check("--manifest", str(m)), 1)


if __name__ == "__main__":
    unittest.main()
