"""Tests for scripts/llamacpp_pin.py (run: python -I -m unittest scripts.test_llamacpp_pin)."""
from __future__ import annotations

import hashlib
import io
import json
import os
import re
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


def make_raw_tarball(path: Path, entries: list[tuple[str, bytes | None]]) -> None:
    """Entries are (full member name, data); data None makes a directory entry."""
    with tarfile.open(path, "w:gz") as tf:
        for name, data in entries:
            if data is None:
                ti = tarfile.TarInfo(name)
                ti.type = tarfile.DIRTYPE
                tf.addfile(ti)
            else:
                ti = tarfile.TarInfo(name)
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

    def _genuine(self, top: str = "llama.cpp-x") -> list[tuple[str, bytes | None]]:
        return [(f"{top}/", None)] + [(f"{top}/{n}", d) for n, d in FILES.items()]

    def test_top_level_file_fails(self) -> None:
        make_raw_tarball(self.tarball, self._genuine() + [("README.md", b"evil" + bytes([10])
        )])
        self._write_manifest()
        self.assertEqual(self.run_verify(), 1)

    def test_second_top_level_dir_fails(self) -> None:
        make_raw_tarball(self.tarball, self._genuine() + self._genuine("evil"))
        self._write_manifest()
        self.assertEqual(self.run_verify(), 1)

    def test_duplicate_entry_fails(self) -> None:
        make_raw_tarball(self.tarball, self._genuine() + [("llama.cpp-x/src/a.c", FILES["src/a.c"])])
        self._write_manifest()
        self.assertEqual(self.run_verify(), 1)

    def test_single_top_dir_with_dir_entries_passes(self) -> None:
        make_raw_tarball(self.tarball, self._genuine() + [("llama.cpp-x/src/", None)])
        self._write_manifest()
        self.assertEqual(self.run_verify(), 0)

    def test_worktree_path_as_git_dir_passes(self) -> None:
        self.git_dir = self.repo
        self.assertEqual(self.run_verify(), 0)

    def test_commit_must_match_manifest(self) -> None:
        write_manifest(self.manifest, "1" * 40, pin.sha256_file(self.tarball))
        self.assertEqual(self.run_verify(), 1)

    def test_unknown_commit_is_usage_error(self) -> None:
        bad = "2" * 40
        write_manifest(self.manifest, bad, pin.sha256_file(self.tarball))
        self.commit = bad
        self.assertEqual(self.run_verify(), 2)


CMAKE_FILE = REPO_ROOT / "src" / "backends" / "llamacpp" / "CMakeLists.txt"
PIN_BEGIN = "# --- pin resolution begin"
PIN_END = "# --- pin resolution end ---"
MIRROR = "file:///D:/mirror/bA.tar.gz"
SHA_A, SHA_B = "a" * 64, "b" * 64


def pin_manifest(path: Path, tag: str, sha: str) -> None:
    path.write_text(json.dumps({
        "schema": 1, "default_track": "mainline",
        "tracks": {"mainline": {
            "tag": tag, "commit": "c" * 40, "sha256": sha,
            "url": f"https://example.invalid/archive/{tag}.tar.gz"}},
    }), encoding="utf-8")


class CMakePinResolutionTests(unittest.TestCase):
    """Configure a mini project made from the pin-resolution block of the real
    src/backends/llamacpp/CMakeLists.txt (no download, no compiler)."""

    @classmethod
    def setUpClass(cls) -> None:
        if shutil.which("cmake") is None:
            raise unittest.SkipTest("cmake not available")
        text = CMAKE_FILE.read_text(encoding="utf-8").replace("\r\n", "\n")
        a, b = text.index(PIN_BEGIN), text.index(PIN_END)
        assert a < b, "pin resolution markers missing from the llamacpp CMakeLists"
        cls.block = text[a:b]

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        (self.tmp / "src").mkdir()
        (self.tmp / "src" / "CMakeLists.txt").write_text(
            "cmake_minimum_required(VERSION 3.20)\nproject(pin_mini LANGUAGES NONE)\n"
            "set(SONDER_WITH_LLAMA_CPP ON)\n" + self.block, encoding="utf-8")
        self.manifest = self.tmp / "pin.json"
        self.build = self.tmp / "build"

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def configure(self, *defs: str) -> tuple[str, str, str]:
        r = subprocess.run(
            ["cmake", "-S", str(self.tmp / "src"), "-B", str(self.build),
             f"-DSONDER_LLAMACPP_PIN_FILE={self.manifest.as_posix()}", *defs],
            capture_output=True, text=True, timeout=120)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        m = re.search(r"llama\.cpp pin: track=\S+ tag=(\S+) url=(\S+) sha256=(\S+)", r.stdout)
        self.assertIsNotNone(m, r.stdout)
        return m.group(1), m.group(2), m.group(3)

    def test_manifest_defaults(self) -> None:
        pin_manifest(self.manifest, "bA", SHA_A)
        self.assertEqual(self.configure(), ("bA", "https://example.invalid/archive/bA.tar.gz", SHA_A))

    def test_url_override_survives_explicit_manifest_sha(self) -> None:
        pin_manifest(self.manifest, "bA", SHA_A)
        self.assertEqual(self.configure(f"-DSONDER_LLAMACPP_URL={MIRROR}", f"-DSONDER_LLAMACPP_SHA256={SHA_A}"),
                         ("bA", MIRROR, SHA_A))
        # Reconfigure in the same build directory (URL is now only in the cache).
        self.assertEqual(self.configure(), ("bA", MIRROR, SHA_A))

    def test_stale_applied_values_move_to_new_manifest(self) -> None:
        pin_manifest(self.manifest, "bA", SHA_A)
        self.configure()
        pin_manifest(self.manifest, "bB", SHA_B)
        # What an old configure left in the cache: its tag, derived URL and SHA.
        self.assertEqual(
            self.configure("-DSONDER_LLAMACPP_TAG=bA",
                           "-DSONDER_LLAMACPP_URL=https://example.invalid/archive/bA.tar.gz",
                           f"-DSONDER_LLAMACPP_SHA256={SHA_A}"),
            ("bB", "https://example.invalid/archive/bB.tar.gz", SHA_B))

    def test_url_override_survives_manifest_bump(self) -> None:
        pin_manifest(self.manifest, "bA", SHA_A)
        self.configure(f"-DSONDER_LLAMACPP_URL={MIRROR}")
        pin_manifest(self.manifest, "bB", SHA_B)
        self.assertEqual(self.configure(), ("bB", MIRROR, SHA_B))


FEATURES_BEGIN = "# --- expected features begin"
FEATURES_END = "# --- expected features end ---"
MTP = "SONDER_LLAMACPP_HAS_MTP"
F8 = "SONDER_LLAMACPP_HAS_GGML_F8_E4M3"
# Defines the real pin must keep detecting (MTP was silently lost once when the
# NextN API moved from src/llama-ext.h to llama.h).
REQUIRED_DEFAULT_FEATURES = (
    MTP,
    "SONDER_LLAMACPP_HAS_NEXTN_LAYER_OFFSET",
    "SONDER_LLAMACPP_HAS_CTX_OTHER",
    "SONDER_LLAMACPP_HAS_MOE_CACHE_SIZE",
    F8,
)


def features_manifest(path: Path, features: object, sha: str = SHA_A) -> None:
    path.write_text(json.dumps({
        "schema": 1, "default_track": "mainline",
        "tracks": {"mainline": {
            "tag": "bA", "commit": "c" * 40, "sha256": sha,
            "url": "https://example.invalid/archive/bA.tar.gz",
            "expected_features": features}},
    }), encoding="utf-8")


class CMakeExpectedFeaturesTests(unittest.TestCase):
    """Configure a mini project made from the pin-resolution and expected-features
    blocks of the real CMakeLists; TEST_FEATURE_DEFS stands in for what header
    detection produced."""

    @classmethod
    def setUpClass(cls) -> None:
        if shutil.which("cmake") is None:
            raise unittest.SkipTest("cmake not available")
        text = CMAKE_FILE.read_text(encoding="utf-8").replace("\r\n", "\n")
        a, b = text.index(PIN_BEGIN), text.index(PIN_END)
        c, d = text.index(FEATURES_BEGIN), text.index(FEATURES_END)
        assert a < b < c < d, "pin / expected-features markers missing or out of order"
        cls.pin_block, cls.features_block = text[a:b], text[c:d]

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        (self.tmp / "src").mkdir()
        (self.tmp / "src" / "CMakeLists.txt").write_text(
            "cmake_minimum_required(VERSION 3.20)\nproject(features_mini LANGUAGES NONE)\n"
            "set(SONDER_WITH_LLAMA_CPP ON)\n" + self.pin_block
            + 'set(SONDER_LLAMACPP_FEATURE_DEFS "${TEST_FEATURE_DEFS}")\n'
            + self.features_block, encoding="utf-8")
        self.manifest = self.tmp / "pin.json"
        self.build = self.tmp / "build"

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def configure(self, defs: list[str], *extra: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["cmake", "-S", str(self.tmp / "src"), "-B", str(self.build),
             f"-DSONDER_LLAMACPP_PIN_FILE={self.manifest.as_posix()}",
             "-DTEST_FEATURE_DEFS=" + ";".join(f"{x}=1" for x in defs), *extra],
            capture_output=True, text=True, timeout=120)

    def test_all_expected_detected_passes(self) -> None:
        features_manifest(self.manifest, [MTP, F8])
        r = self.configure([MTP, F8, "SONDER_LLAMACPP_HAS_KV_UNIFIED"])
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("expected_features present", r.stdout)

    def test_missing_expected_feature_fails_configure(self) -> None:
        features_manifest(self.manifest, [MTP, F8])
        r = self.configure([F8])
        self.assertNotEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn(MTP, r.stderr)
        self.assertNotIn(f"produce {F8}", r.stderr)

    def test_prefix_of_a_define_does_not_count(self) -> None:
        features_manifest(self.manifest, ["SONDER_LLAMACPP_HAS_MT"])
        r = self.configure([MTP])
        self.assertNotEqual(r.returncode, 0, r.stdout + r.stderr)

    def test_empty_or_absent_list_is_not_enforced(self) -> None:
        features_manifest(self.manifest, [])
        self.assertEqual(self.configure([]).returncode, 0)
        pin_manifest(self.manifest, "bA", SHA_A)
        r = self.configure([])
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)

    def test_sha_override_is_not_enforced(self) -> None:
        features_manifest(self.manifest, [MTP])
        r = self.configure([], "-DSONDER_LLAMACPP_TAG=bB", f"-DSONDER_LLAMACPP_SHA256={SHA_B}")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("not enforced: SHA-256 override", r.stdout)

    def test_local_source_dir_is_not_enforced(self) -> None:
        features_manifest(self.manifest, [MTP])
        r = self.configure([], f"-DFETCHCONTENT_SOURCE_DIR_LLAMACPP={self.tmp.as_posix()}")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("not enforced: local source dir", r.stdout)

    def test_malformed_entries_fail_configure(self) -> None:
        for bad in (["HAS_MTP"], ["SONDER_LLAMACPP_HAS_MTP=1"], "SONDER_LLAMACPP_HAS_MTP"):
            with self.subTest(bad=bad):
                shutil.rmtree(self.build, ignore_errors=True)
                features_manifest(self.manifest, bad)
                r = self.configure([MTP])
                self.assertNotEqual(r.returncode, 0, r.stdout + r.stderr)

    def test_default_manifest_requires_mtp_and_friends(self) -> None:
        _, entry = pin.load_entry(pin.DEFAULT_MANIFEST, None)
        for name in REQUIRED_DEFAULT_FEATURES:
            self.assertIn(name, entry.get("expected_features", []))


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
        self.assertRegex(entry["tag"], r"^(b\d+|v\d+\.\d+\.\d+)$")

    def test_changed_manifest_tag_is_detected(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            m = Path(d) / "pin.json"
            _, entry = pin.load_entry(pin.DEFAULT_MANIFEST, None)
            write_manifest(m, entry["commit"], entry["sha256"], tag="b99999")
            self.assertEqual(self.run_check("--manifest", str(m)), 1)

    def test_changed_manifest_release_tag_is_detected(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            m = Path(d) / "pin.json"
            _, entry = pin.load_entry(pin.DEFAULT_MANIFEST, None)
            write_manifest(m, entry["commit"], entry["sha256"], tag="v9.9.9")
            self.assertEqual(self.run_check("--manifest", str(m)), 1)

    @staticmethod
    def _docs_root(d: Path, license_review: str, integration: str) -> Path:
        (d / "docs" / "integration").mkdir(parents=True)
        (d / "docs" / "LICENSE_REVIEW.md").write_text(license_review, encoding="utf-8")
        (d / "docs" / "integration" / "llamacpp.md").write_text(integration, encoding="utf-8")
        return d

    def test_release_tags_count_only_in_llamacpp_scope(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            m = Path(d) / "pin.json"
            write_manifest(m, "a" * 40, "b" * 64, tag="v1.0.0")
            root = self._docs_root(
                Path(d) / "ok",
                "## Other\n- cpp-httplib v0.58.0\n"
                "### llama.cpp / GGML\n- revision/tag: `v1.0.0`\n"
                "### doctest\n- revision/tag: v2.5.3\n",
                "llama.cpp is pinned at `v1.0.0`.\n")
            self.assertEqual(self.run_check("--manifest", str(m), "--root", str(root)), 0)

    def test_stale_release_tag_in_llamacpp_section_is_detected(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            m = Path(d) / "pin.json"
            write_manifest(m, "a" * 40, "b" * 64, tag="v1.0.0")
            root = self._docs_root(
                Path(d) / "rel",
                "### llama.cpp / GGML\n- revision/tag: `v0.9.0`\n",
                "llama.cpp is pinned at `v1.0.0`.\n")
            self.assertEqual(self.run_check("--manifest", str(m), "--root", str(root)), 1)

    def test_stale_build_tag_under_release_manifest_is_detected(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            m = Path(d) / "pin.json"
            write_manifest(m, "a" * 40, "b" * 64, tag="v1.0.0")
            root = self._docs_root(
                Path(d) / "build",
                "### llama.cpp / GGML\n- revision/tag: `v1.0.0`\n",
                "Offline builds: /path/to/llama.cpp-b11195\n")
            self.assertEqual(self.run_check("--manifest", str(m), "--root", str(root)), 1)

    def test_changed_manifest_sha_is_detected(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            m = Path(d) / "pin.json"
            _, entry = pin.load_entry(pin.DEFAULT_MANIFEST, None)
            write_manifest(m, entry["commit"], "f" * 64, tag=entry["tag"])
            self.assertEqual(self.run_check("--manifest", str(m)), 1)


if __name__ == "__main__":
    unittest.main()
