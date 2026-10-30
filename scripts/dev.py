"""Configure, build, and test with tools installed in the selected Python environment."""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import cmake

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["configure", "build", "test"], default="test", nargs="?")
    args = parser.parse_args()
    environment = os.environ.copy()
    paths = [str(Path(sys.executable).parent), cmake.CMAKE_BIN_DIR]
    user_presets = ROOT / "CMakeUserPresets.json"
    preset = "release"
    if user_presets.exists():
        preset = "local"
        compiler = json.loads(user_presets.read_text())["configurePresets"][0]["cacheVariables"]["CMAKE_CXX_COMPILER"]
        paths.insert(0, str(Path(compiler).parent))
    environment["PATH"] = os.pathsep.join(paths + [environment.get("PATH", "")])
    executable = str(Path(cmake.CMAKE_BIN_DIR) / ("cmake.exe" if os.name == "nt" else "cmake"))

    def run(command):
        subprocess.run(command, cwd=ROOT, env=environment, check=True)

    run([executable, "--preset", preset, f"-DPython3_EXECUTABLE={sys.executable}"])
    if args.action != "configure":
        run([executable, "--build", "build", "--config", "Release"])
    if args.action == "test":
        run([str(Path(cmake.CMAKE_BIN_DIR) / ("ctest.exe" if os.name == "nt" else "ctest")),
             "--test-dir", "build", "-C", "Release", "--output-on-failure"])


if __name__ == "__main__":
    main()
