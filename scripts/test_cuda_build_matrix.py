"""Tests for MAIN-03 CUDA build matrix, configure checks, CUB-10445 guard, and SASS verification.

Run: python -I -m unittest scripts.test_cuda_build_matrix -v
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PRESETS_FILE = REPO_ROOT / "CMakePresets.json"
CUDA_CHECK_CMAKE = REPO_ROOT / "cmake" / "SonderCudaCheck.cmake"
LLAMACPP_CMAKE = REPO_ROOT / "src" / "backends" / "llamacpp" / "CMakeLists.txt"


def _write_mock_toolkit(
    root: Path,
    major: int,
    minor: int,
    patch: int,
    build: int = 0,
    version_json_cublas: str | None = None,
    raw_version_json: str | None = None,
) -> Path:
    inc = root / "include"
    inc.mkdir(parents=True, exist_ok=True)
    (inc / "cublas_api.h").write_text(
        "#pragma once\n"
        f"#define CUBLAS_VER_MAJOR {major}\n"
        f"#define CUBLAS_VER_MINOR {minor}\n"
        f"#define CUBLAS_VER_PATCH {patch}\n"
        f"#define CUBLAS_VER_BUILD {build}\n",
        encoding="utf-8",
    )
    if raw_version_json is not None:
        (root / "version.json").write_text(raw_version_json, encoding="utf-8")
    elif version_json_cublas is not None:
        (root / "version.json").write_text(
            json.dumps({"libcublas": {"name": "CUDA cuBLAS", "version": version_json_cublas}}),
            encoding="utf-8",
        )
    return root


class CMakePresetsMatrixTests(unittest.TestCase):
    def setUp(self) -> None:
        self.doc = json.loads(PRESETS_FILE.read_text(encoding="utf-8"))
        self.cfg = {p["name"]: p for p in self.doc["configurePresets"]}
        self.bld = {p["name"]: p for p in self.doc["buildPresets"]}
        self.tst = {p["name"]: p for p in self.doc["testPresets"]}

    def test_blackwell_120a_and_120f_presets_present_and_paired(self) -> None:
        expected = {
            "windows-cuda-blackwell": ("msvc-release", "120a"),
            "linux-cuda-blackwell": ("linux-release", "120a"),
            "windows-cuda-blackwell-120f": ("windows-cuda-blackwell", "120f"),
            "linux-cuda-blackwell-120f": ("linux-cuda-blackwell", "120f"),
            "windows-cuda-rtx": ("msvc-release", "86;89;120a"),
            "linux-cuda-rtx": ("linux-release", "86;89;120a"),
        }
        for name, (parent, archs) in expected.items():
            with self.subTest(preset=name):
                self.assertIn(name, self.cfg)
                self.assertIn(name, self.bld)
                self.assertIn(name, self.tst)
                self.assertEqual(self.cfg[name]["inherits"], parent)
                self.assertEqual(self.cfg[name]["cacheVariables"]["CMAKE_CUDA_ARCHITECTURES"], archs)
                self.assertEqual(self.bld[name]["configurePreset"], name)
                self.assertEqual(self.tst[name]["configurePreset"], name)

    def test_blackwell_presets_keep_graphs_and_fa_quants_enabled(self) -> None:
        for name in ("windows-cuda-blackwell", "linux-cuda-blackwell", "windows-cuda-rtx", "linux-cuda-rtx"):
            cv = self.cfg[name]["cacheVariables"]
            self.assertEqual(cv["SONDER_WITH_LLAMA_CPP"], "ON")
            self.assertEqual(cv["GGML_CUDA"], "ON")
            self.assertEqual(cv["GGML_CUDA_GRAPHS"], "ON")
            self.assertEqual(cv["GGML_CUDA_FA_ALL_QUANTS"], "ON")

    def test_no_preset_uses_bare_120_target(self) -> None:
        for name, p in self.cfg.items():
            archs = p.get("cacheVariables", {}).get("CMAKE_CUDA_ARCHITECTURES", "")
            for tok in filter(None, archs.split(";")):
                self.assertNotRegex(
                    tok,
                    r"^12[0-9](-real|-virtual)?$",
                    f"preset {name} uses bare Blackwell architecture {tok!r}",
                )

    def test_cmake_accepts_presets_file(self) -> None:
        cmake = shutil.which("cmake")
        if cmake is None:
            raise unittest.SkipTest("cmake not on PATH")
        r = subprocess.run(
            [cmake, "--list-presets"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        expected = "windows-cuda-blackwell-120f" if os.name == "nt" else "linux-cuda-blackwell-120f"
        self.assertIn(expected, r.stdout)

    def test_docs_agree_with_cuda_preset_matrix(self) -> None:
        stale_re = re.compile(r"CMAKE_CUDA_ARCHITECTURES\s*=\s*\"?(?:86;89;120|120)(?![0-9af])")
        md_paths = [REPO_ROOT / "README.md", *sorted((REPO_ROOT / "docs").rglob("*.md")), *sorted((REPO_ROOT / "src").rglob("*.md"))]
        for md in md_paths:
            text = md.read_text(encoding="utf-8")
            m = stale_re.search(text)
            self.assertIsNone(m, f"stale bare-120 CMAKE_CUDA_ARCHITECTURES doc in {md.relative_to(REPO_ROOT)}: {m.group(0) if m else ''}")

        placement = (REPO_ROOT / "docs" / "PLACEMENT.md").read_text(encoding="utf-8")
        for token in ("CMAKE_CUDA_ARCHITECTURES=86;89;120a", "sm_120a", "windows-cuda-blackwell-120f", "CUDA >= 13.4"):
            self.assertIn(token, placement)


class CMakeCudaCheckConfigureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cmake = shutil.which("cmake")
        if cls.cmake is None:
            raise unittest.SkipTest("cmake not available")

    def setUp(self) -> None:
        self.assertTrue(CUDA_CHECK_CMAKE.is_file(), f"missing {CUDA_CHECK_CMAKE}")
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.src = self.tmp / "src"
        self.src.mkdir()
        (self.src / "CMakeLists.txt").write_text(
            "cmake_minimum_required(VERSION 3.21)\n"
            "project(cuda_check_unit LANGUAGES NONE)\n"
            f'list(APPEND CMAKE_MODULE_PATH "{(REPO_ROOT / "cmake").as_posix()}")\n'
            "include(SonderCudaCheck)\n"
            "sonder_check_cuda_config()\n"
            'message(STATUS "GRAPHS_AFTER=${GGML_CUDA_GRAPHS}")\n',
            encoding="utf-8",
        )
        self._build_idx = 0

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def configure(self, *defs: str) -> subprocess.CompletedProcess[str]:
        self._build_idx += 1
        bdir = self.tmp / f"b{self._build_idx}"
        return subprocess.run(
            [self.cmake, "-S", str(self.src), "-B", str(bdir), *defs],
            capture_output=True,
            text=True,
            timeout=60,
        )

    def test_cuda_13_5_and_13_4_pass_and_default_graphs_on(self) -> None:
        tk135 = _write_mock_toolkit(self.tmp / "tk135", 13, 10, 0, 0, "13.10.0.0")
        r135 = self.configure(
            f"-DSONDER_CUDA_CHECK_MOCK_TOOLKIT_ROOT={tk135.as_posix()}",
            "-DSONDER_CUDA_CHECK_MOCK_VERSION=13.5.0",
            "-DCMAKE_CUDA_ARCHITECTURES=120a",
        )
        self.assertEqual(r135.returncode, 0, r135.stdout + r135.stderr)
        self.assertIn("preferred >= 13.5 met", r135.stdout)
        self.assertIn("GRAPHS_AFTER=ON", r135.stdout)

        tk134 = _write_mock_toolkit(self.tmp / "tk134", 13, 7, 0, 27, "13.7.0.27")
        r134 = self.configure(
            f"-DSONDER_CUDA_CHECK_MOCK_TOOLKIT_ROOT={tk134.as_posix()}",
            "-DSONDER_CUDA_CHECK_MOCK_VERSION=13.4.1",
            "-DCMAKE_CUDA_ARCHITECTURES=86;89;120a",
        )
        self.assertEqual(r134.returncode, 0, r134.stdout + r134.stderr)
        self.assertIn("13.5+ preferred", r134.stdout)
        self.assertIn("cuBLAS=13.7.0.27", r134.stdout)

    def test_graphs_explicit_off_warns(self) -> None:
        tk135 = _write_mock_toolkit(self.tmp / "tk135", 13, 10, 0, 0, "13.10.0.0")
        r = self.configure(
            f"-DSONDER_CUDA_CHECK_MOCK_TOOLKIT_ROOT={tk135.as_posix()}",
            "-DSONDER_CUDA_CHECK_MOCK_VERSION=13.5.0",
            "-DCMAKE_CUDA_ARCHITECTURES=120a",
            "-DGGML_CUDA_GRAPHS=OFF",
        )
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("GGML_CUDA_GRAPHS is OFF", r.stderr)

    def test_cub_10445_cublas_13_6_0_is_refused_in_header_and_version_json(self) -> None:
        tk_hdr = _write_mock_toolkit(self.tmp / "tk_hdr", 13, 6, 0, 2, "13.6.0.2")
        r1 = self.configure(
            f"-DSONDER_CUDA_CHECK_MOCK_TOOLKIT_ROOT={tk_hdr.as_posix()}",
            "-DSONDER_CUDA_CHECK_MOCK_VERSION=13.4.0",
            "-DCMAKE_CUDA_ARCHITECTURES=120a",
        )
        self.assertNotEqual(r1.returncode, 0)
        self.assertIn("CUB-10445", r1.stderr)

        # Even when cublas_api.h says 13.6.1, a version.json still at 13.6.0 fails closed.
        tk_vj = _write_mock_toolkit(self.tmp / "tk_vj", 13, 6, 1, 5, "13.6.0.2")
        r2 = self.configure(
            f"-DSONDER_CUDA_CHECK_MOCK_TOOLKIT_ROOT={tk_vj.as_posix()}",
            "-DSONDER_CUDA_CHECK_MOCK_VERSION=13.4.0",
            "-DCMAKE_CUDA_ARCHITECTURES=120a",
        )
        self.assertNotEqual(r2.returncode, 0)
        self.assertIn("CUB-10445", r2.stderr)

        # Patched 13.6.1 passes when legacy CUDA 13.3.1 is explicitly allowed.
        tk_fixed = _write_mock_toolkit(self.tmp / "tk_fixed", 13, 6, 1, 5, "13.6.1.5")
        r3 = self.configure(
            f"-DSONDER_CUDA_CHECK_MOCK_TOOLKIT_ROOT={tk_fixed.as_posix()}",
            "-DSONDER_CUDA_CHECK_MOCK_VERSION=13.3.1",
            "-DSONDER_ALLOW_LEGACY_CUDA=ON",
            "-DCMAKE_CUDA_ARCHITECTURES=120a",
        )
        self.assertEqual(r3.returncode, 0, r3.stdout + r3.stderr)

    def test_architecture_and_cmake_version_guards_and_hostile_inputs(self) -> None:
        tk135 = _write_mock_toolkit(self.tmp / "tk135", 13, 10, 0, 0, "13.10.0.0")
        bad_cases = [
            (["-DCMAKE_CUDA_ARCHITECTURES=120"], "bare Blackwell target"),
            (["-DCMAKE_CUDA_ARCHITECTURES=86;121-virtual"], "bare Blackwell target"),
            (["-DCMAKE_CUDA_ARCHITECTURES=120f", "-DSONDER_CUDA_CHECK_CMAKE_VERSION=3.31.6"], "CMake 3.31.6 rejects"),
            (["-DCMAKE_CUDA_ARCHITECTURES=120f", "-DSONDER_CUDA_CHECK_CMAKE_VERSION=4.0.0"], "CMake 4.0.0 rejects"),
            (
                [
                    "-DCMAKE_CUDA_ARCHITECTURES=120f",
                    "-DSONDER_CUDA_CHECK_CMAKE_VERSION=4.0.2",
                    "-DSONDER_CUDA_CHECK_MOCK_VERSION=12.8.1",
                    "-DSONDER_ALLOW_LEGACY_CUDA=ON",
                ],
                ">= 12.9",
            ),
            (
                [
                    "-DCMAKE_CUDA_ARCHITECTURES=120a",
                    "-DSONDER_CUDA_CHECK_MOCK_VERSION=12.7.0",
                    "-DSONDER_ALLOW_LEGACY_CUDA=ON",
                ],
                ">= 12.8",
            ),
            (["-DCMAKE_CUDA_ARCHITECTURES=61"], "removed in CUDA 13"),
            (["-DCMAKE_CUDA_ARCHITECTURES="], "must not be empty"),
            (["-DCMAKE_CUDA_ARCHITECTURES=;;;"], "must not be empty"),
            (["-DCMAKE_CUDA_ARCHITECTURES=-120"], "invalid CUDA architecture"),
            (["-DCMAKE_CUDA_ARCHITECTURES=999999"], "out-of-range"),
            (["-DCMAKE_CUDA_ARCHITECTURES=../120a"], "unsafe"),
            (["-DCMAKE_CUDA_ARCHITECTURES=native;120a"], "cannot be combined"),
        ]
        for extra_args, needle in bad_cases:
            with self.subTest(args=extra_args):
                base_args = [f"-DSONDER_CUDA_CHECK_MOCK_TOOLKIT_ROOT={tk135.as_posix()}"]
                if not any(a.startswith("-DSONDER_CUDA_CHECK_MOCK_VERSION=") for a in extra_args):
                    base_args.append("-DSONDER_CUDA_CHECK_MOCK_VERSION=13.5.0")
                r = self.configure(*base_args, *extra_args)
                self.assertNotEqual(r.returncode, 0, r.stdout + r.stderr)
                norm_err = " ".join(r.stderr.split())
                self.assertIn(needle, norm_err)

    def test_corrupt_version_json_and_missing_cublas_header_fail_closed(self) -> None:
        tk_corrupt = _write_mock_toolkit(self.tmp / "tk_corrupt", 13, 7, 0, 0, raw_version_json="{truncated")
        r1 = self.configure(
            f"-DSONDER_CUDA_CHECK_MOCK_TOOLKIT_ROOT={tk_corrupt.as_posix()}",
            "-DSONDER_CUDA_CHECK_MOCK_VERSION=13.4.1",
            "-DCMAKE_CUDA_ARCHITECTURES=120a",
        )
        self.assertNotEqual(r1.returncode, 0)
        self.assertIn("version.json", r1.stderr)

        empty_tk = self.tmp / "tk_empty"
        (empty_tk / "include").mkdir(parents=True)
        r2 = self.configure(
            f"-DSONDER_CUDA_CHECK_MOCK_TOOLKIT_ROOT={empty_tk.as_posix()}",
            "-DSONDER_CUDA_CHECK_MOCK_VERSION=13.4.1",
            "-DCMAKE_CUDA_ARCHITECTURES=120a",
        )
        self.assertNotEqual(r2.returncode, 0)
        self.assertIn("cublas_api.h", r2.stderr)

    def test_llamacpp_cmakelists_wires_cuda_check_before_fetchcontent(self) -> None:
        text = LLAMACPP_CMAKE.read_text(encoding="utf-8")
        i_inc = text.find("include(SonderCudaCheck)")
        i_call = text.find("sonder_check_cuda_config()")
        i_fetch = text.find("FetchContent_MakeAvailable(llamacpp)")
        self.assertGreater(i_inc, 0)
        self.assertGreater(i_call, i_inc)
        self.assertGreater(i_fetch, i_call)


class HostCudaAndRealSassTests(unittest.TestCase):
    """Exercises real CUDAToolkit discovery and real nvcc + cuobjdump SASS checks when present."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.cmake = shutil.which("cmake")
        if cls.cmake is None:
            raise unittest.SkipTest("cmake not available")

    def _find_cuda_bin_tool(self, name: str) -> str | None:
        found = shutil.which(name)
        if found:
            return found
        cuda_path = os.environ.get("CUDA_PATH")
        if cuda_path:
            exe = Path(cuda_path) / "bin" / (f"{name}.exe" if os.name == "nt" else name)
            if exe.is_file():
                return str(exe)
        return None

    def test_host_cudatoolkit_configure_when_present(self) -> None:
        nvcc = self._find_cuda_bin_tool("nvcc")
        if nvcc is None:
            raise unittest.SkipTest("CUDA Toolkit not installed on this host")
        self.assertTrue(CUDA_CHECK_CMAKE.is_file(), f"missing {CUDA_CHECK_CMAKE}")
        ver_out = subprocess.check_output([nvcc, "--version"], text=True, errors="replace")
        m = re.search(r"release (\d+)\.(\d+)", ver_out)
        if not m or (int(m.group(1)), int(m.group(2))) < (13, 4):
            raise unittest.SkipTest(f"Host nvcc older than 13.4: {ver_out.strip()}")

        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            src = tmp / "src"
            src.mkdir()
            (src / "CMakeLists.txt").write_text(
                "cmake_minimum_required(VERSION 3.21)\n"
                "project(host_cuda_probe LANGUAGES NONE)\n"
                f'list(APPEND CMAKE_MODULE_PATH "{(REPO_ROOT / "cmake").as_posix()}")\n'
                "include(SonderCudaCheck)\n"
                "sonder_check_cuda_config()\n",
                encoding="utf-8",
            )
            r = subprocess.run(
                [self.cmake, "-S", str(src), "-B", str(tmp / "build"), "-DCMAKE_CUDA_ARCHITECTURES=120a"],
                capture_output=True,
                text=True,
                timeout=60,
            )
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            self.assertIn("CUB-10445 clean", r.stdout)

    def test_real_nvcc_and_cuobjdump_sass_verification_when_present(self) -> None:
        nvcc = self._find_cuda_bin_tool("nvcc")
        cuobjdump = self._find_cuda_bin_tool("cuobjdump")
        if nvcc is None or cuobjdump is None:
            raise unittest.SkipTest("nvcc/cuobjdump not installed on this host")
        self.assertTrue(CUDA_CHECK_CMAKE.is_file(), f"missing {CUDA_CHECK_CMAKE}")

        env = dict(os.environ)
        ccbin_args: list[str] = []
        if os.name == "nt" and shutil.which("cl.exe") is None:
            vswhere = Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")) / (
                r"Microsoft Visual Studio\Installer\vswhere.exe"
            )
            if not vswhere.is_file():
                raise unittest.SkipTest("cl.exe not on PATH and vswhere.exe not found")
            cl_paths = subprocess.check_output(
                [
                    str(vswhere),
                    "-latest",
                    "-products",
                    "*",
                    "-requires",
                    "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
                    "-find",
                    r"VC\Tools\MSVC\*\bin\Hostx64\x64\cl.exe",
                ],
                text=True,
            ).splitlines()
            if not cl_paths:
                raise unittest.SkipTest("MSVC Hostx64/x64/cl.exe not found")
            ccbin_args = ["-ccbin", str(Path(cl_paths[0]).parent)]

        kernel_src = (
            'extern "C" __global__ void k_fp4(float* out, int a0, int a1, int a2, int a3, '
            "int b0, int b1, unsigned sa, unsigned sb) {\n"
            "  float d0=0,d1=0,d2=0,d3=0;\n"
            "#if __CUDA_ARCH__ >= 1200\n"
            '  asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.'
            'e2m1.e2m1.f32.ue4m3 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3},%10,{0,0},%11,{0,0};" '
            ': "+f"(d0),"+f"(d1),"+f"(d2),"+f"(d3) '
            ': "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1),"r"(sa),"r"(sb));\n'
            "#endif\n"
            "  out[0]=d0+d1+d2+d3;\n"
            "}\n"
        )
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            cu = tmp / "probe.cu"
            cu.write_text(kernel_src, encoding="utf-8")

            # Bare sm_120 must fail ptxas assembly on block-scaled FP4 MMA.
            r_bare = subprocess.run(
                [nvcc, *ccbin_args, "-arch=sm_120", "-cubin", str(cu), "-o", str(tmp / "bare.cubin")],
                env=env,
                capture_output=True,
                text=True,
                timeout=60,
            )
            self.assertNotEqual(r_bare.returncode, 0, "bare sm_120 unexpectedly assembled block-scaled FP4 MMA")

            # Both sm_120a and sm_120f must assemble and pass SonderCudaCheck SASS verification.
            for arch in ("sm_120a", "sm_120f"):
                with self.subTest(arch=arch):
                    cubin = tmp / f"{arch}.cubin"
                    r_nvcc = subprocess.run(
                        [nvcc, *ccbin_args, f"-arch={arch}", "-cubin", str(cu), "-o", str(cubin)],
                        env=env,
                        capture_output=True,
                        text=True,
                        timeout=60,
                    )
                    self.assertEqual(r_nvcc.returncode, 0, r_nvcc.stdout + r_nvcc.stderr)
                    r_sass = subprocess.run(
                        [
                            self.cmake,
                            "-DSONDER_CUDA_VERIFY_SASS=ON",
                            f"-DTARGET_BINARY={cubin.as_posix()}",
                            f"-DCUOBJDUMP_EXECUTABLE={Path(cuobjdump).as_posix()}",
                            "-P",
                            str(CUDA_CHECK_CMAKE),
                        ],
                        capture_output=True,
                        text=True,
                        timeout=60,
                    )
                    self.assertEqual(r_sass.returncode, 0, r_sass.stdout + r_sass.stderr)
                    self.assertIn("native Blackwell SASS verified", r_sass.stdout)


if __name__ == "__main__":
    unittest.main()
