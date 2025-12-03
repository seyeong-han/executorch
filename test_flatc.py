import subprocess
import os
import sys


# Mocking the _run_flatc from _flatbuffer.py
def test_run_flatc():
    flatc_path = "flatc"
    args = ["--version"]
    try:
        subprocess.run([flatc_path] + list(args), check=True, capture_output=True)
        print("flatc --version passed")
    except subprocess.CalledProcessError as e:
        print("flatc --version failed")
        print(e.stdout.decode())
        print(e.stderr.decode())

    # Test failure case
    args = ["--invalid-arg"]
    try:
        subprocess.run([flatc_path] + list(args), check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        print("Caught expected failure:")
        print(f"STDOUT: {e.stdout.decode(errors='replace')}")
        print(f"STDERR: {e.stderr.decode(errors='replace')}")


if __name__ == "__main__":
    test_run_flatc()
