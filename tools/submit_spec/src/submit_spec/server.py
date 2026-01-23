"""
MCP Server implementation for submit_spec tool.

This server provides two tools for agents:
1. submit_spec - Submit a TLA+ specification for validation
2. get_submission_status - Check remaining submission attempts
"""

import asyncio
import json
import os
from datetime import datetime
from typing import Optional

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

from .evaluator import SpecEvaluator, EvaluationResult


class SubmitSpecServer:
    """
    MCP Server for TLA+ specification submission and validation.

    This server manages submission attempts and delegates evaluation
    to the SpecEvaluator class.
    """

    def __init__(self):
        """Initialize the server with configuration from environment variables."""
        # Read configuration from environment
        self.task_name = os.environ.get("SYSMOBENCH_TASK")
        self.spec_module = os.environ.get("SYSMOBENCH_SPEC_MODULE")
        self.max_attempts = int(os.environ.get("SYSMOBENCH_MAX_ATTEMPTS", "3"))
        self.output_dir = os.environ.get("SYSMOBENCH_OUTPUT", "output")
        self.syntax_timeout = int(os.environ.get("SYSMOBENCH_SYNTAX_TIMEOUT", "30"))
        self.runtime_timeout = int(os.environ.get("SYSMOBENCH_RUNTIME_TIMEOUT", "300"))

        # Validate required configuration
        if not self.task_name:
            raise ValueError("SYSMOBENCH_TASK environment variable is required")
        if not self.spec_module:
            raise ValueError("SYSMOBENCH_SPEC_MODULE environment variable is required")

        # Generate timestamp for this run session
        self.run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Initialize evaluator
        self.evaluator = SpecEvaluator(
            task_name=self.task_name,
            spec_module=self.spec_module,
            output_dir=self.output_dir,
            run_timestamp=self.run_timestamp,
            syntax_timeout=self.syntax_timeout,
            runtime_timeout=self.runtime_timeout,
        )

        # Track submission state (protected by _lock for thread safety)
        self.attempt_count = 0
        self.results: list[EvaluationResult] = []
        self._lock = asyncio.Lock()

        # Create MCP server
        self.server = Server("submit_spec")
        self._register_handlers()

    def _register_handlers(self):
        """Register tool handlers with the MCP server."""

        @self.server.list_tools()
        async def list_tools() -> list[Tool]:
            """Return the list of available tools."""
            return [
                Tool(
                    name="submit_spec",
                    description=(
                        "Submit a TLA+ specification for validation. "
                        "The specification will be validated in two phases: "
                        "Phase 1 checks syntax using SANY parser, "
                        "Phase 2 runs TLC model checking. "
                        f"Maximum {self.max_attempts} attempts allowed."
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "spec_content": {
                                "type": "string",
                                "description": "The complete TLA+ specification content",
                            },
                            "config_content": {
                                "type": "string",
                                "description": "The TLC configuration file content",
                            },
                        },
                        "required": ["spec_content", "config_content"],
                    },
                ),
                Tool(
                    name="get_submission_status",
                    description=(
                        "Get the current submission status including "
                        "remaining attempts and previous results summary."
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {},
                        "required": [],
                    },
                ),
            ]

        @self.server.call_tool()
        async def call_tool(name: str, arguments: dict) -> list[TextContent]:
            """Handle tool calls."""
            if name == "submit_spec":
                return await self._handle_submit_spec(arguments)
            elif name == "get_submission_status":
                return await self._handle_get_status(arguments)
            else:
                return [TextContent(
                    type="text",
                    text=json.dumps({"error": f"Unknown tool: {name}"}),
                )]

    async def _handle_submit_spec(self, arguments: dict) -> list[TextContent]:
        """
        Handle the submit_spec tool call.

        Args:
            arguments: Tool arguments containing spec_content and config_content

        Returns:
            List containing a TextContent with the evaluation result

        Note:
            This method uses a lock to ensure thread-safe state management
            in case of concurrent calls.
        """
        async with self._lock:
            # Check if max attempts reached
            if self.attempt_count >= self.max_attempts:
                return [TextContent(
                    type="text",
                    text=json.dumps({
                        "error": "Maximum submission attempts reached",
                        "max_attempts": self.max_attempts,
                        "attempts_used": self.attempt_count,
                    }),
                )]

            # Validate arguments
            spec_content = arguments.get("spec_content")
            config_content = arguments.get("config_content")

            if not spec_content:
                return [TextContent(
                    type="text",
                    text=json.dumps({"error": "spec_content is required"}),
                )]

            if not config_content:
                return [TextContent(
                    type="text",
                    text=json.dumps({"error": "config_content is required"}),
                )]

            # Increment attempt counter
            self.attempt_count += 1
            current_attempt = self.attempt_count

        # Run evaluation outside the lock (this is the expensive operation)
        result = self.evaluator.evaluate(
            spec_content=spec_content,
            config_content=config_content,
            attempt=current_attempt,
            max_attempts=self.max_attempts,
        )

        # Acquire lock again to update results
        async with self._lock:
            self.results.append(result)
            self.evaluator.save_summary(self.results)

        return [TextContent(
            type="text",
            text=json.dumps(result.to_dict(), indent=2),
        )]

    async def _handle_get_status(self, arguments: dict) -> list[TextContent]:
        """
        Handle the get_submission_status tool call.

        Returns:
            List containing a TextContent with the current status

        Note:
            This method uses a lock to ensure consistent state reading.
        """
        async with self._lock:
            status = {
                "task_name": self.task_name,
                "spec_module": self.spec_module,
                "max_attempts": self.max_attempts,
                "attempts_used": self.attempt_count,
                "remaining_attempts": self.max_attempts - self.attempt_count,
                "run_timestamp": self.run_timestamp,
                "output_directory": str(self.evaluator.run_dir),
                "previous_results": [
                    {
                        "attempt": r.attempt,
                        "success": r.success,
                        "phase1_success": r.phase1_syntax.success if r.phase1_syntax else None,
                        "phase2_success": r.phase2_runtime.success if r.phase2_runtime else None,
                    }
                    for r in self.results
                ],
            }

        return [TextContent(
            type="text",
            text=json.dumps(status, indent=2),
        )]

    async def run(self):
        """Run the MCP server."""
        async with stdio_server() as (read_stream, write_stream):
            await self.server.run(
                read_stream,
                write_stream,
                self.server.create_initialization_options(),
            )


def create_server() -> SubmitSpecServer:
    """Factory function to create a SubmitSpecServer instance."""
    return SubmitSpecServer()
