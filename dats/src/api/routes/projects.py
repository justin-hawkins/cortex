"""
Project management API routes.

Handles project CRUD operations and configuration.
"""

import logging
import uuid
from datetime import datetime
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, status

from src.api.dependencies import get_provenance_tracker, verify_api_key
from src.api.schemas import (
    ConstitutionSetRequest,
    ProjectConfig,
    ProjectCreateRequest,
    ProjectListResponse,
    ProjectResponse,
    ProjectUpdateRequest,
    TaskMode,
)
from src.config.settings import get_settings

logger = logging.getLogger(__name__)

router = APIRouter()

# In-memory project storage (TODO: Move to storage/projects.py)
_projects: dict[str, dict[str, Any]] = {}


def _get_project_store() -> dict[str, dict[str, Any]]:
    """Get the project store."""
    global _projects
    return _projects


def _init_default_project():
    """Initialize the default project if it doesn't exist."""
    settings = get_settings()
    store = _get_project_store()
    
    if settings.default_project not in store:
        store[settings.default_project] = {
            "id": settings.default_project,
            "name": settings.default_project,
            "description": "Default project",
            "config": {
                "qa_profile": "consensus",
                "default_mode": TaskMode.AUTONOMOUS.value,
                "constitution": None,
            },
            "repo_url": None,
            "created_at": datetime.utcnow(),
            "updated_at": None,
        }


@router.post(
    "/projects",
    response_model=ProjectResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a project",
    description="Create a new project",
)
async def create_project(
    request: ProjectCreateRequest,
    api_key: Optional[str] = Depends(verify_api_key),
):
    """Create a new project."""
    store = _get_project_store()
    
    # Generate project ID from name
    project_id = request.name.lower().replace(" ", "-")
    
    if project_id in store:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "code": "PROJECT_EXISTS",
                "message": f"Project with name '{request.name}' already exists",
            },
        )
    
    # Build config
    config = request.config or ProjectConfig()
    
    project = {
        "id": project_id,
        "name": request.name,
        "description": request.description,
        "config": config.model_dump(),
        "repo_url": request.repo_url,
        "created_at": datetime.utcnow(),
        "updated_at": None,
    }
    
    store[project_id] = project
    logger.info(f"Created project: {project_id}")
    
    return ProjectResponse(
        id=project_id,
        name=request.name,
        description=request.description,
        config=config,
        repo_url=request.repo_url,
        created_at=project["created_at"],
        updated_at=None,
        task_count=0,
    )


@router.get(
    "/projects",
    response_model=ProjectListResponse,
    summary="List projects",
    description="List all projects",
)
async def list_projects(
    api_key: Optional[str] = Depends(verify_api_key),
):
    """List all projects."""
    _init_default_project()
    store = _get_project_store()
    provenance_tracker = get_provenance_tracker()
    
    projects = []
    for project_id, project in store.items():
        # Count tasks for this project
        records = provenance_tracker.get_by_project(project_id)
        task_ids = set(r.task_id for r in records)
        
        projects.append(ProjectResponse(
            id=project_id,
            name=project["name"],
            description=project.get("description"),
            config=ProjectConfig(**project.get("config", {})),
            repo_url=project.get("repo_url"),
            created_at=project["created_at"],
            updated_at=project.get("updated_at"),
            task_count=len(task_ids),
        ))
    
    return ProjectListResponse(
        projects=projects,
        total=len(projects),
    )


@router.get(
    "/projects/{project_id}",
    response_model=ProjectResponse,
    summary="Get project",
    description="Get project details",
)
async def get_project(
    project_id: str,
    api_key: Optional[str] = Depends(verify_api_key),
):
    """Get project details."""
    _init_default_project()
    store = _get_project_store()
    provenance_tracker = get_provenance_tracker()
    
    if project_id not in store:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "code": "PROJECT_NOT_FOUND",
                "message": f"Project with ID '{project_id}' not found",
            },
        )
    
    project = store[project_id]
    
    # Count tasks for this project
    records = provenance_tracker.get_by_project(project_id)
    task_ids = set(r.task_id for r in records)
    
    return ProjectResponse(
        id=project_id,
        name=project["name"],
        description=project.get("description"),
        config=ProjectConfig(**project.get("config", {})),
        repo_url=project.get("repo_url"),
        created_at=project["created_at"],
        updated_at=project.get("updated_at"),
        task_count=len(task_ids),
    )


@router.put(
    "/projects/{project_id}",
    response_model=ProjectResponse,
    summary="Update project",
    description="Update project configuration",
)
async def update_project(
    project_id: str,
    request: ProjectUpdateRequest,
    api_key: Optional[str] = Depends(verify_api_key),
):
    """Update project configuration."""
    store = _get_project_store()
    provenance_tracker = get_provenance_tracker()
    
    if project_id not in store:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "code": "PROJECT_NOT_FOUND",
                "message": f"Project with ID '{project_id}' not found",
            },
        )
    
    project = store[project_id]
    
    # Update fields
    if request.description is not None:
        project["description"] = request.description
    if request.config is not None:
        project["config"] = request.config.model_dump()
    
    project["updated_at"] = datetime.utcnow()
    
    logger.info(f"Updated project: {project_id}")
    
    # Count tasks for this project
    records = provenance_tracker.get_by_project(project_id)
    task_ids = set(r.task_id for r in records)
    
    return ProjectResponse(
        id=project_id,
        name=project["name"],
        description=project.get("description"),
        config=ProjectConfig(**project.get("config", {})),
        repo_url=project.get("repo_url"),
        created_at=project["created_at"],
        updated_at=project["updated_at"],
        task_count=len(task_ids),
    )


@router.delete(
    "/projects/{project_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete project",
    description="Delete a project",
)
async def delete_project(
    project_id: str,
    api_key: Optional[str] = Depends(verify_api_key),
):
    """Delete a project."""
    settings = get_settings()
    store = _get_project_store()
    
    if project_id not in store:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "code": "PROJECT_NOT_FOUND",
                "message": f"Project with ID '{project_id}' not found",
            },
        )
    
    # Don't allow deleting the default project
    if project_id == settings.default_project:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "code": "CANNOT_DELETE_DEFAULT",
                "message": "Cannot delete the default project",
            },
        )
    
    del store[project_id]
    logger.info(f"Deleted project: {project_id}")


@router.post(
    "/projects/{project_id}/constitution",
    response_model=ProjectResponse,
    summary="Set constitution",
    description="Set the constitution/standards for a project",
)
async def set_constitution(
    project_id: str,
    request: ConstitutionSetRequest,
    api_key: Optional[str] = Depends(verify_api_key),
):
    """Set the constitution/standards for a project."""
    store = _get_project_store()
    provenance_tracker = get_provenance_tracker()
    
    if project_id not in store:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "code": "PROJECT_NOT_FOUND",
                "message": f"Project with ID '{project_id}' not found",
            },
        )
    
    project = store[project_id]
    
    # Update constitution in config
    if "config" not in project:
        project["config"] = {}
    project["config"]["constitution"] = request.constitution
    project["updated_at"] = datetime.utcnow()
    
    logger.info(f"Set constitution for project: {project_id}")
    
    # Count tasks for this project
    records = provenance_tracker.get_by_project(project_id)
    task_ids = set(r.task_id for r in records)
    
    return ProjectResponse(
        id=project_id,
        name=project["name"],
        description=project.get("description"),
        config=ProjectConfig(**project.get("config", {})),
        repo_url=project.get("repo_url"),
        created_at=project["created_at"],
        updated_at=project["updated_at"],
        task_count=len(task_ids),
    )