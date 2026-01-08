"""
Worker module for DATS.

Provides specialized workers for different task domains.
"""

from src.workers.base import BaseWorker
from src.workers.code_general import CodeGeneralWorker
from src.workers.code_vision import CodeVisionWorker
from src.workers.code_embedded import CodeEmbeddedWorker
from src.workers.documentation import DocumentationWorker
from src.workers.ui_design import UIDesignWorker

__all__ = [
    "BaseWorker",
    "CodeGeneralWorker",
    "CodeVisionWorker",
    "CodeEmbeddedWorker",
    "DocumentationWorker",
    "UIDesignWorker",
]