"""
Pipeline module for DATS.

Provides the main orchestration logic for processing user requests
through the agent pipeline.
"""

from src.pipeline.orchestrator import AgentPipeline, PipelineResult

__all__ = ["AgentPipeline", "PipelineResult"]