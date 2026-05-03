"""
TV evaluation helper tools.

Only genuinely reusable mechanical code goes here.
Per-system logic (window generation, TV module, mappings) is written
by the evaluation agent per the tv-eval skill guide.
"""

from .runner import run_tv_batch, check_one_window, summarize

__all__ = ["run_tv_batch", "check_one_window", "summarize"]
