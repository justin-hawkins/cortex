"""
Monitoring commands for DATS CLI.

Handles workers, queues, failures, and provenance export.
"""

from pathlib import Path
from typing import Optional

import httpx
import typer

from src.cli.output import Formatter


def workers(
    ctx: typer.Context,
):
    """
    View worker status.
    
    Examples:
        dats workers
    """
    formatter: Formatter = ctx.obj["formatter"]
    api_url: str = ctx.obj["api_url"]
    api_key: Optional[str] = ctx.obj["api_key"]
    
    headers = {}
    if api_key:
        headers["X-API-Key"] = api_key
    
    try:
        response = httpx.get(
            f"{api_url}/api/v1/workers",
            headers=headers,
            timeout=30.0,
        )
        
        if response.status_code == 200:
            data = response.json()
            formatter.print_workers(data.get("workers", []))
            formatter.console.print(f"\n[dim]Active workers: {data.get('total_active', 0)}[/dim]")
        else:
            error = response.json().get("error", {})
            formatter.error(
                error.get("message", "Failed to get worker status"),
                code=error.get("code"),
            )
            raise typer.Exit(1)
            
    except httpx.RequestError as e:
        formatter.error(f"Connection error: {e}")
        raise typer.Exit(1)


def queues(
    ctx: typer.Context,
):
    """
    View queue depths.
    
    Examples:
        dats queues
    """
    formatter: Formatter = ctx.obj["formatter"]
    api_url: str = ctx.obj["api_url"]
    api_key: Optional[str] = ctx.obj["api_key"]
    
    headers = {}
    if api_key:
        headers["X-API-Key"] = api_key
    
    try:
        response = httpx.get(
            f"{api_url}/api/v1/queues",
            headers=headers,
            timeout=30.0,
        )
        
        if response.status_code == 200:
            data = response.json()
            formatter.print_queues(data.get("queues", []))
        else:
            error = response.json().get("error", {})
            formatter.error(
                error.get("message", "Failed to get queue status"),
                code=error.get("code"),
            )
            raise typer.Exit(1)
            
    except httpx.RequestError as e:
        formatter.error(f"Connection error: {e}")
        raise typer.Exit(1)


def failures(
    ctx: typer.Context,
    last: int = typer.Option(
        10,
        "--last", "-n",
        help="Number of failures to show",
    ),
):
    """
    View recent failures.
    
    Examples:
        dats failures
        dats failures --last 20
    """
    formatter: Formatter = ctx.obj["formatter"]
    api_url: str = ctx.obj["api_url"]
    api_key: Optional[str] = ctx.obj["api_key"]
    
    headers = {}
    if api_key:
        headers["X-API-Key"] = api_key
    
    try:
        response = httpx.get(
            f"{api_url}/api/v1/failures",
            headers=headers,
            params={"last": last},
            timeout=30.0,
        )
        
        if response.status_code == 200:
            data = response.json()
            formatter.print_failures(data.get("failures", []))
        else:
            error = response.json().get("error", {})
            formatter.error(
                error.get("message", "Failed to get failures"),
                code=error.get("code"),
            )
            raise typer.Exit(1)
            
    except httpx.RequestError as e:
        formatter.error(f"Connection error: {e}")
        raise typer.Exit(1)


def provenance_export(
    ctx: typer.Context,
    task_id: str = typer.Argument(
        ...,
        help="Task ID to export provenance for",
    ),
    format: str = typer.Option(
        "json",
        "--format", "-f",
        help="Export format (json)",
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output", "-o",
        help="Output file path",
    ),
):
    """
    Export provenance for a task.
    
    Examples:
        dats provenance export <task-id>
        dats provenance export <task-id> --output provenance.json
    """
    import json
    
    formatter: Formatter = ctx.obj["formatter"]
    api_url: str = ctx.obj["api_url"]
    api_key: Optional[str] = ctx.obj["api_key"]
    
    headers = {}
    if api_key:
        headers["X-API-Key"] = api_key
    
    try:
        response = httpx.get(
            f"{api_url}/api/v1/provenance/{task_id}/export",
            headers=headers,
            params={"format": format},
            timeout=30.0,
        )
        
        if response.status_code == 200:
            data = response.json()
            
            if output:
                with open(output, "w") as f:
                    json.dump(data, f, indent=2, default=str)
                formatter.success(f"Provenance exported to {output}")
            else:
                formatter.print_json(data)
        elif response.status_code == 404:
            formatter.error(f"Task not found: {task_id}")
            raise typer.Exit(1)
        else:
            error = response.json().get("error", {})
            formatter.error(
                error.get("message", "Failed to export provenance"),
                code=error.get("code"),
            )
            raise typer.Exit(1)
            
    except httpx.RequestError as e:
        formatter.error(f"Connection error: {e}")
        raise typer.Exit(1)