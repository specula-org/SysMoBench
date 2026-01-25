#!/usr/bin/env python3
"""
Entry point for the submit_spec MCP server.

This script starts the MCP server that allows agents to submit TLA+ specifications
for validation. The server reads configuration from environment variables:

Required:
    SYSMOBENCH_TASK: Name of the evaluation task
    SYSMOBENCH_SPEC_MODULE: TLA+ module name expected in the specification

Optional:
    SYSMOBENCH_MAX_ATTEMPTS: Maximum submission attempts (default: 3)
    SYSMOBENCH_OUTPUT: Output directory for logs (default: output)
    SYSMOBENCH_SYNTAX_TIMEOUT: Syntax validation timeout in seconds (default: 30)
    SYSMOBENCH_RUNTIME_TIMEOUT: TLC runtime timeout in seconds (default: 300)

Usage:
    export SYSMOBENCH_TASK="spin"
    export SYSMOBENCH_SPEC_MODULE="specTrace"
    python mcp_server.py
"""

import asyncio
import io
import os
import sys
from pathlib import Path

# Suppress stdout during module imports (MCP requires clean stdout for JSON-RPC)
# Some modules print debug info on import which breaks MCP protocol
_real_stdout = sys.stdout
sys.stdout = io.StringIO()

# Add the src directory to the Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from submit_spec import create_server

# Restore stdout after imports
sys.stdout = _real_stdout


def main():
    """Main entry point."""
    try:
        server = create_server()
        asyncio.run(server.run())
    except ValueError as e:
        print(f"Configuration error: {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nServer stopped.", file=sys.stderr)
        sys.exit(0)
    except Exception as e:
        print(f"Server error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
