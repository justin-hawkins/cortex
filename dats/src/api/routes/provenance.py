"""
Provenance API routes.

Handles provenance queries, lineage tracking, and impact analysis.
"""

import json
import logging
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status

from src.api.dependencies import get_provenance_tracker, verify_api_key
from src.api.schemas import (
    DependenciesResponse,
    DependencyInfo,
    DependentsResponse,
    ImpactAnalysisResponse,
    ProvenanceRecord,
    ProvenanceResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get(
    "/provenance/{artifact_id}",
    response_model=ProvenanceResponse,
    summary="Get provenance record",
    description="Get provenance record for an artifact",
)
async def get_provenance(
    artifact_id: str,
    api_key: Optional[str] = Depends(verify_api_key),
):
    """Get provenance record for an artifact."""
    provenance_tracker = get_provenance_tracker()
    
    # Find the record that produced this artifact
    record = provenance_tracker.get_producer(artifact_id)
    
    if not record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "code": "ARTIFACT_NOT_FOUND",
                "message": f"No provenance record found for artifact {artifact_id}",
            },
        )
    
    return ProvenanceResponse(
        record=ProvenanceRecord(
            id=record.id,
            task_id=record.task_id,
            project_id=record.project_id,
            model_used=record.model_used or record.execution.model,
            worker_id=record.worker_id or record.execution.worker_id,
            created_at=record.created_at or datetime.utcnow(),
            verification_status=record.verification_status,
            is_tainted=record.is_tainted(),
        )
    )


@router.get(
    "/provenance/{artifact_id}/dependents",
    response_model=DependentsResponse,
    summary="Get artifact dependents",
    description="Get all artifacts that depend on this artifact (forward traversal)",
)
async def get_dependents(
    artifact_id: str,
    transitive: bool = True,
    api_key: Optional[str] = Depends(verify_api_key),
):
    """Get all artifacts that depend on this artifact."""
    provenance_tracker = get_provenance_tracker()
    
    # Use the provenance graph for traversal
    dependent_ids = provenance_tracker.graph.find_dependents(
        artifact_id,
        transitive=transitive,
    )
    
    # Build dependency info for each dependent
    dependents = []
    for dep_id in dependent_ids:
        prov_id = provenance_tracker.graph.get_provenance_for_artifact(dep_id)
        dependents.append(DependencyInfo(
            artifact_id=dep_id,
            provenance_id=prov_id or "",
            relationship="consumed",
        ))
    
    return DependentsResponse(
        artifact_id=artifact_id,
        dependents=dependents,
        total=len(dependents),
    )


@router.get(
    "/provenance/{artifact_id}/dependencies",
    response_model=DependenciesResponse,
    summary="Get artifact dependencies",
    description="Get all artifacts this artifact depends on (backward traversal)",
)
async def get_dependencies(
    artifact_id: str,
    transitive: bool = True,
    api_key: Optional[str] = Depends(verify_api_key),
):
    """Get all artifacts this artifact depends on."""
    provenance_tracker = get_provenance_tracker()
    
    # Use the provenance graph for traversal
    dependency_ids = provenance_tracker.graph.find_dependencies(
        artifact_id,
        transitive=transitive,
    )
    
    # Build dependency info for each dependency
    dependencies = []
    for dep_id in dependency_ids:
        prov_id = provenance_tracker.graph.get_provenance_for_artifact(dep_id)
        dependencies.append(DependencyInfo(
            artifact_id=dep_id,
            provenance_id=prov_id or "",
            relationship="consumed",
        ))
    
    return DependenciesResponse(
        artifact_id=artifact_id,
        dependencies=dependencies,
        total=len(dependencies),
    )


@router.get(
    "/provenance/{artifact_id}/impact",
    response_model=ImpactAnalysisResponse,
    summary="Impact analysis",
    description="Analyze impact if this artifact were tainted",
)
async def analyze_impact(
    artifact_id: str,
    api_key: Optional[str] = Depends(verify_api_key),
):
    """Analyze impact if this artifact were tainted."""
    provenance_tracker = get_provenance_tracker()
    
    # Use the impact analysis from provenance tracker
    report = provenance_tracker.impact_analysis(artifact_id)
    
    return ImpactAnalysisResponse(
        artifact_id=artifact_id,
        direct_dependents=report.direct_dependents,
        transitive_dependents=report.transitive_dependents,
        total_affected=report.total_affected,
        max_depth=report.max_depth,
    )


@router.get(
    "/provenance/{artifact_id}/export",
    summary="Export provenance",
    description="Export provenance record in specified format",
)
async def export_provenance(
    artifact_id: str,
    format: str = "json",
    api_key: Optional[str] = Depends(verify_api_key),
):
    """Export provenance record in specified format."""
    provenance_tracker = get_provenance_tracker()
    
    # Find the record that produced this artifact
    record = provenance_tracker.get_producer(artifact_id)
    
    if not record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "code": "ARTIFACT_NOT_FOUND",
                "message": f"No provenance record found for artifact {artifact_id}",
            },
        )
    
    # Get lineage
    lineage = provenance_tracker.get_lineage(artifact_id)
    
    # Build export data
    export_data = {
        "artifact_id": artifact_id,
        "provenance": record.to_dict(),
        "lineage": [r.to_dict() for r in lineage],
        "exported_at": datetime.utcnow().isoformat(),
    }
    
    if format.lower() == "json":
        return export_data
    else:
        # Default to JSON
        return export_data