"""Ordinary failed-start receipts at the real stress CLI boundary (Linux)."""
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class FailureReceiptTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="sonder-stress-control-")
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.driver = Path(os.environ.get("SONDER_STRESS_DRIVER", Path(__file__).with_name("stress_mock.py")))

    def failure(self, binary, completed):
        receipt = self.root / "receipt.json"
        arguments = [sys.executable, str(self.driver), "--binary", str(binary), "--cycles", "3", "--out", str(receipt)]
        # Legacy-driver red checks can keep its default delay; CI controls use
        # the longer drain window explicitly, just like the positive gate.
        delay = os.environ.get("SONDER_STRESS_DELAY_MS")
        if delay is not None:
            arguments += ["--mock-delay-ms", delay]
        run = subprocess.run(
            arguments,
            capture_output=True, text=True, timeout=30,
        )
        self.assertNotEqual(run.returncode, 0)
        self.assertIn("Traceback", run.stderr)
        self.assertTrue(receipt.is_file(), "failed qualification did not retain a JSON receipt")
        result = json.loads(receipt.read_text())
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["completed_cycles"], completed)
        self.assertEqual(result["total_requests"], 68 * completed)
        self.assertEqual(result["failure"]["type"], "AssertionError")
        self.assertIn("server exited before readiness", result["failure"]["message"])
        return result

    def test_early_exit_is_not_a_success_or_missing_receipt(self):
        self.failure("/usr/bin/true", 0)

    def test_failed_restart_preserves_completed_real_server_cycle(self):
        binary = Path(os.environ.get("SONDER_STRESS_BINARY", "build/ci-linux/sonder-infer")).resolve()
        self.assertTrue(binary.is_file(), "build the actual mock server before running this control")
        wrapper = self.root / "exit-on-second-start.py"
        marker = self.root / "first-start"
        wrapper.write_text(
            "#!" + sys.executable + "\nimport os,sys\nfrom pathlib import Path\n"
            f"marker=Path({str(marker)!r})\n"
            "if not marker.exists():\n    marker.touch()\n"
            f"    binary={str(binary)!r}\n    os.execv(binary,[binary,*sys.argv[1:]])\n"
        )
        wrapper.chmod(0o700)
        result = self.failure(wrapper, 1)
        cycle = result["cycles"][0]
        self.assertEqual(cycle["exit_code"], 0)
        self.assertTrue(cycle["ready_removed"])
        self.assertEqual(cycle["shutdown_health_in_flight"], 4)


if __name__ == "__main__":
    unittest.main()
