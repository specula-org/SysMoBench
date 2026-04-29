"""
TLA+ Tools Setup Script

Downloads `tla2tools.jar` and `CommunityModules-deps.jar` into `lib/`.
Runtime path resolution lives in `tla_eval.utils.setup_utils`.
"""

import shutil
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


def main():
    print_status("TLA+ Tools Setup")
    print_status("================")

    try:
        print_status("\n=== Checking Runtime Dependencies ===")
        if not check_java_available():
            print_warning("Java not found. TLA+ tooling requires Java to run.")
            print_status("Please install Java 11+ and ensure it is in your PATH.")
        else:
            print_success("Java is available")

        print_status("\n=== Setting Up Tools ===")
        setup_tla_tools()

        print_status("\n=== Verification ===")
        if verify_tools():
            print_success("\n✓ Setup completed successfully!")
            print_status("\nTool paths:")
            print_status(f"  tla2tools.jar: {get_tla_tools_path()}")
            print_status(f"  CommunityModules-deps.jar: {get_community_modules_path()}")
        else:
            print_warning("\n⚠ Setup completed with warnings")

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
