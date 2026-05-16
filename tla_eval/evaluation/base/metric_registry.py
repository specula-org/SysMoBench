"""
Central registry of all evaluation metrics.

Each metric maps a CLI-visible name (e.g. `compilation_check`) to an evaluator
class. `create_evaluator(name, **kwargs)` instantiates one with the right
default parameters merged in.
"""

from typing import Dict, List, Any, Type


class MetricInfo:
    """Static info about an evaluation metric."""

    def __init__(self,
                 name: str,
                 description: str,
                 evaluator_class: Type,
                 default_params: Dict[str, Any] = None):
        self.name = name
        self.description = description
        self.evaluator_class = evaluator_class
        self.default_params = default_params or {}


class MetricRegistry:
    """Central registry for all available evaluation metrics."""

    def __init__(self):
        self._metrics: Dict[str, MetricInfo] = {}
        self._metrics_registered = False

    def register_metric(self, metric_info: MetricInfo):
        self._metrics[metric_info.name] = metric_info

    def _ensure_registered(self):
        if not self._metrics_registered:
            self._register_default_metrics()
            self._metrics_registered = True

    def get_metric(self, name: str) -> MetricInfo:
        self._ensure_registered()
        if name not in self._metrics:
            raise ValueError(f"Unknown metric: {name}")
        return self._metrics[name]

    def list_metrics(self) -> List[MetricInfo]:
        self._ensure_registered()
        return sorted(self._metrics.values(), key=lambda m: m.name)

    def _register_default_metrics(self):
        from ..syntax.compilation_check import CompilationCheckEvaluator
        from ..syntax.action_decomposition import ActionDecompositionEvaluator
        from ..semantics.runtime_check import RuntimeCheckEvaluator
        from ..semantics.manual_invariant_evaluator import ManualInvariantEvaluator
        from ..semantics.coverage_evaluator import CoverageEvaluator
        from ..semantics.runtime_coverage_evaluator import RuntimeCoverageEvaluator
        from ..semantics.transition_validation import TransitionValidationEvaluator

        self.register_metric(MetricInfo(
            name="compilation_check",
            description="Basic TLA+ compilation checking using SANY parser",
            evaluator_class=CompilationCheckEvaluator,
        ))

        self.register_metric(MetricInfo(
            name="action_decomposition",
            description="Evaluate individual actions separately for better granularity",
            evaluator_class=ActionDecompositionEvaluator,
            default_params={"validation_timeout": 30, "keep_temp_files": False},
        ))

        self.register_metric(MetricInfo(
            name="runtime_check",
            description="Model checking with TLC using the spec's own invariants",
            evaluator_class=RuntimeCheckEvaluator,
        ))

        self.register_metric(MetricInfo(
            name="invariant_verification",
            description="Verify the spec against expert-written invariants translated to its variables",
            evaluator_class=ManualInvariantEvaluator,
        ))

        self.register_metric(MetricInfo(
            name="coverage",
            description="TLA+ specification coverage analysis using TLC coverage statistics",
            evaluator_class=CoverageEvaluator,
            default_params={"tlc_timeout": 60, "coverage_interval": 1},
        ))

        self.register_metric(MetricInfo(
            name="runtime_coverage",
            description="Runtime coverage using simulation mode to identify successful vs error-prone actions",
            evaluator_class=RuntimeCoverageEvaluator,
            default_params={
                "num_simulations": 20,
                "simulation_depth": 50,
                "traces_per_simulation": 50,
                "tlc_timeout": 30,
                "coverage_interval": 1,
            },
        ))

        self.register_metric(MetricInfo(
            name="transition_validation",
            description="Per-action conformance to captured system traces (agent-driven, costs ~$1-4 per spec)",
            evaluator_class=TransitionValidationEvaluator,
            default_params={"tv_budget": 5.0, "tv_timeout": 1800},
        ))


_registry = MetricRegistry()


def get_metric_registry() -> MetricRegistry:
    return _registry


def get_available_metrics() -> List[str]:
    return [m.name for m in get_metric_registry().list_metrics()]


def create_evaluator(metric_name: str, language: str = "TLA+", **kwargs):
    """Instantiate the evaluator for `metric_name`, merging default params with overrides."""
    metric_info = get_metric_registry().get_metric(metric_name)

    # Some evaluators accept `validation_timeout` instead of `tlc_timeout`.
    if metric_name in ("action_decomposition", "compilation_check") and "tlc_timeout" in kwargs:
        kwargs["validation_timeout"] = kwargs.pop("tlc_timeout")

    params = {**metric_info.default_params, **kwargs}

    # Forward `language` only to evaluators whose __init__ accepts it. The
    # legacy ones (action_decomposition, coverage, runtime_coverage) don't.
    import inspect
    init_params = inspect.signature(metric_info.evaluator_class.__init__).parameters
    if "language" in init_params:
        params["language"] = language

    return metric_info.evaluator_class(**params)
