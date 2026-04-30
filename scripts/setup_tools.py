"""
SysMoBench setup script.

Downloads `tla2tools.jar` and `CommunityModules-deps.jar` into `lib/`,
then audits the host for the system-level prerequisites required by the
benchmark and reports anything missing.

Runtime path resolution lives in `tla_eval.utils.setup_utils`.
"""

import shutil
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent

sys.path.insert(0, str(PROJECT_ROOT))
from tla_eval.utils.setup_utils import (
    get_tla_tools_path,
    get_community_modules_path,
    check_java_available,
    validate_tla_tools_setup,
)

TLA_TOOLS_URL = "https://github.com/tlaplus/tlaplus/releases/download/v1.8.0/tla2tools.jar"
COMMUNITY_MODULES_URL = "https://github.com/tlaplus/CommunityModules/releases/download/202505152026/CommunityModules-deps.jar"


def print_status(message: str):
    logger.info(message)


def print_success(message: str):
    logger.info(f"✓ {message}")


def print_warning(message: str):
    logger.warning(f"⚠ {message}")


def print_error(message: str):
    logger.error(f"✗ {message}")


def download_file(url: str, output_path: Path) -> bool:
    """Download `url` to `output_path` with a progress indicator."""
    try:
        print_status(f"Downloading {output_path.name}...")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            with urllib.request.urlopen(url) as response:
                file_size = int(response.headers.get('Content-Length', 0))
                downloaded = 0
                chunk_size = 8192
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    temp_file.write(chunk)
                    downloaded += len(chunk)
                    if file_size > 0:
                        progress = (downloaded / file_size) * 100
                        print(f"\rProgress: {progress:.1f}%", end='', flush=True)
                print()

            shutil.move(temp_file.name, str(output_path))
            print_success(f"{output_path.name} downloaded successfully")
            return True

    except Exception as e:
        print_error(f"Failed to download {output_path.name}: {e}")
        return False


def setup_tla_tools() -> bool:
    """Download tla2tools.jar and CommunityModules-deps.jar if missing."""
    print_status("Setting up TLA+ tools...")

    (PROJECT_ROOT / "lib").mkdir(exist_ok=True)
    success = True

    tla_tools_path = get_tla_tools_path()
    if not tla_tools_path.exists():
        if not download_file(TLA_TOOLS_URL, tla_tools_path):
            success = False
    else:
        print_success("tla2tools.jar already exists")

    community_modules_path = get_community_modules_path()
    if not community_modules_path.exists():
        if not download_file(COMMUNITY_MODULES_URL, community_modules_path):
            print_warning("CommunityModules-deps.jar download failed - this is optional for basic functionality")
    else:
        print_success("CommunityModules-deps.jar already exists")

    return success


def verify_tools() -> bool:
    """Print a final readiness summary."""
    print_status("Verifying TLA+ tools installation...")
    results = validate_tla_tools_setup()

    if results["java_available"]:
        print_success(f"Java available: {results['java_version'] or 'version detected'}")
    else:
        print_warning("Java not found - TLA+ tooling requires Java to run")

    if results["tla_tools_exists"]:
        print_success(f"tla2tools.jar found ({results['tla_tools_size']:,} bytes)")
    else:
        print_error("tla2tools.jar not found or empty")

    if results["community_modules_exists"]:
        print_success(f"CommunityModules-deps.jar found ({results['community_modules_size']:,} bytes)")
    else:
        print_warning("CommunityModules-deps.jar not found - optional for advanced features")

    return results["ready"]


def _probe_version(cmd, timeout=5):
    """Run `cmd` and return its first stdout/stderr line if exit==0, else None."""
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if r.returncode == 0:
            return (r.stdout or r.stderr).strip().splitlines()[0]
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        pass
    return None


# Each row: (label, version-probe command, install hint, affects)
HOST_DEPENDENCIES = [
    ("Docker", ["docker", "--version"],
     "https://docs.docker.com/engine/install/",
     "spin, mutex, rwmutex (Asterinas-based harnesses)"),
    ("Go 1.26+", ["go", "version"],
     "https://go.dev/dl/",
     "etcd"),
    ("Maven", ["mvn", "-v"],
     "apt install maven",
     "zookeeper, redisraft"),
    ("javac (JDK build chain)", ["javac", "--version"],
     "apt install default-jdk",
     "zookeeper, redisraft"),
]


def check_host_dependencies() -> int:
    """Audit host-side prerequisites. Returns the number missing."""
    missing = 0

    py = sys.version_info
    if py >= (3, 8):
        print_success(f"Python: {py.major}.{py.minor}.{py.micro}")
    else:
        print_warning(f"Python {py.major}.{py.minor} is too old — SysMoBench requires Python 3.8+")
        print_status("    Install: https://www.python.org/downloads/ or use pyenv")
        missing += 1

    if check_java_available():
        print_success(f"Java: {_probe_version(['java', '-version']) or 'available'}")
    else:
        print_warning("Java not found — required for SANY and TLC")
        print_status("    Install: apt install openjdk-21-jdk  (or any JDK 11+)")
        missing += 1

    for label, cmd, hint, affects in HOST_DEPENDENCIES:
        v = _probe_version(cmd)
        if v:
            print_success(f"{label}: {v}")
        else:
            print_warning(f"{label} not found — required for: {affects}")
            print_status(f"    Install: {hint}")
            missing += 1

    for cli in ("claude", "codex"):
        v = _probe_version([cli, "--version"])
        if v:
            print_success(f"Coding-agent CLI ({cli}): {v}")
            break
    else:
        print_warning("No coding-agent CLI found — required for transition validation and the agent invariant translator")
        print_status("    Install one of:")
        print_status("      - claude-code: https://github.com/anthropics/claude-code")
        print_status("      - codex:       https://github.com/openai/codex")
        missing += 1

    return missing


def main():
    print_status("SysMoBench setup")
    print_status("================")

    try:
        print_status("\n=== Setting up TLA+ tools ===")
        setup_tla_tools()

        print_status("\n=== Verifying TLA+ tools ===")
        tla_ready = verify_tools()

        print_status("\n=== Checking host dependencies ===")
        missing = check_host_dependencies()

        print()
        if tla_ready and missing == 0:
            print_success("Setup complete — all prerequisites satisfied.")
        elif tla_ready:
            print_warning(f"Setup complete with {missing} missing host dependency item(s).")
            print_status("Install the items above before running the affected systems.")
        else:
            print_error("Setup incomplete: TLA+ tools are not ready.")
            sys.exit(1)

    except KeyboardInterrupt:
        print_error("\nSetup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print_error(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
