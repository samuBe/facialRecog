import subprocess
import sys


def test_runtime_detects_mode_conflict_in_subprocess():
    script = """
from fhe.runtime import claim_toy_fhe, native_fhe_unavailable_reason
print(claim_toy_fhe())
print(native_fhe_unavailable_reason())
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=True,
    )

    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    assert lines[0] == "None"
    assert "toy OpenFHE mode was activated first" in lines[1]
