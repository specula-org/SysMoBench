"""
Metric Registry: Central registry for all evaluation metrics.

This module defines all available evaluation metrics organized by dimension.
"""

from typing import Dict, List, Any, Type
from abc import ABC, abstractmethod


class MetricInfo:
    """Information about a specific metric"""
    
    def __init__(self, 
                 name: str, 
                 dimension: str, 
                 description: str, 
                 evaluator_class: Type,
                 default_params: Dict[str, Any] = None):
        self.name = name
        self.dimension = dimension
        self.description = description
        self.evaluator_class = evaluator_class
        self.default_params = default_params or {}


class MetricRegistry:
    """Central registry for all available evaluation metrics"""
    
    def __init__(self):
        self._metrics: Dict[str, MetricInfo] = {}
        self._metrics_registered = False
    
    def register_metric(self, metric_info: MetricInfo):
        """Register a new metric"""
        self._metrics[metric_info.name] = metric_info
    
    def _ensure_registered(self):
        """Ensure default metrics are registered (lazy loading)"""
        if not self._metrics_registered:
            self._register_default_metrics()
            self._metrics_registered = True
    
    def get_metric(self, name: str) -> MetricInfo:
        """Get metric information by name"""
        self._ensure_registered()
        if name not in self._metrics:
            raise ValueError(f"Unknown metric: {name}")
        return self._metrics[name]
    
    def list_metrics(self, dimension: str = None) -> List[MetricInfo]:
        """List all metrics, optionally filtered by dimension"""
        self._ensure_registered()
        metrics = list(self._metrics.values())
        if dimension:
            metrics = [m for m in metrics if m.dimension == dimension]
        return sorted(metrics, key=lambda m: (m.dimension, m.name))
    
    def list_dimensions(self) -> List[str]:
        """List all available dimensions"""
        self._ensure_registered()
        dimensions = set(m.dimension for m in self._metrics.values())
        return sorted(dimensions)
    
    def _register_default_metrics(self):
        """Register all default metrics"""
        # Import evaluator classes (avoiding circular imports)
        from ..syntax.compilation_check import CompilationCheckEvaluator
        from ..syntax.action_decomposition import ActionDecompositionEvaluator
        from ..semantics.runtime_check import RuntimeCheckEvaluator
        from ..consistency.trace_validation import TraceValidationEvaluator
        from ..composite.composite_evaluation import CompositeEvaluator
        
        # Syntax dimension metrics
        self.register_metric(MetricInfo(
            name="compilation_check",
            dimension="syntax",
            description="Basic TLA+ compilation checking using SANY parser",
            evaluator_class=CompilationCheckEvaluator
        ))
        
        self.register_metric(MetricInfo(
            name="action_decomposition", 
            dimension="syntax",
            description="Evaluate individual actions separately for better granularity",
            evaluator_class=ActionDecompositionEvaluator
        ))
        
        # self.register_metric(MetricInfo(
        #     name="pass_at_k",
        #     dimension="syntax", 
        #     description="Pass@k evaluation from code generation benchmarks",
        #     evaluator_class=PassAtKEvaluator,
        #     default_params={"k": 5}
        # ))
        
        # Semantics dimension metrics
        self.register_metric(MetricInfo(
            name="runtime_check",
            dimension="semantics",
            description="Model checking with TLC using specification's own invariants",
            evaluator_class=RuntimeCheckEvaluator
        ))
        
        # Future semantics metrics (placeholders)
        # self.register_metric(MetricInfo(
        #     name="ast_analysis",
        #     dimension="semantics",
        #     description="Static analysis of specification structure and complexity", 
        #     evaluator_class=ASTAnalysisEvaluator
        # ))
        
        # self.register_metric(MetricInfo(
        #     name="llm_quality_assessment",
        #     dimension="semantics",
        #     description="LLM-based quality evaluation comparing source and spec",
        #     evaluator_class=LLMQualityAssessmentEvaluator
        # ))
        
        # Consistency dimension metrics
        self.register_metric(MetricInfo(
            name="trace_validation",
            dimension="consistency", 
            description="Full trace generation and validation pipeline",
            evaluator_class=TraceValidationEvaluator
        ))
        
        # Composite dimension metrics
        self.register_metric(MetricInfo(
            name="composite",
            dimension="composite",
            description="Integrated evaluation combining action decomposition, compilation check, and runtime check",
            evaluator_class=CompositeEvaluator,
            default_params={"invariant_iterations": 3}
        ))
        
        # Future consistency metrics (placeholders)
        # self.register_metric(MetricInfo(
        #     name="progressive_granularity",
        #     dimension="consistency",
        #     description="Multi-level trace validation with increasing granularity",
        #     evaluator_class=ProgressiveGranularityEvaluator,
        #     default_params={"level": 1}
        # ))
        
        # self.register_metric(MetricInfo(
        #     name="instrumentation_strategy", 
        #     dimension="consistency",
        #     description="Comprehensive instrumentation with selective reduction",
        #     evaluator_class=InstrumentationStrategyEvaluator
        # ))


# Global registry instance
_registry = MetricRegistry()


def get_metric_registry() -> MetricRegistry:
    """Get the global metric registry"""
    return _registry


def get_available_metrics(dimension: str = None) -> List[str]:
    """Get list of available metric names, optionally filtered by dimension"""
    registry = get_metric_registry()
    metrics = registry.list_metrics(dimension)
    return [m.name for m in metrics]


def get_available_dimensions() -> List[str]:
    """Get list of available evaluation dimensions"""
    registry = get_metric_registry()
    return registry.list_dimensions()


def create_evaluator(metric_name: str, **kwargs):
    """Create an evaluator instance for the given metric"""
    registry = get_metric_registry()
    metric_info = registry.get_metric(metric_name)
    
    # Merge default params with provided kwargs
    params = {**metric_info.default_params, **kwargs}
    
    return metric_info.evaluator_class(**params)