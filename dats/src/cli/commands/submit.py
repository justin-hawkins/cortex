"""
Submit command for DATS CLI.

Handles submitting work requests to the system.
"""

from pathlib import Path
from typing import Optional

import httpx
import typer

from src.cli.output import Formatter, get_progress_spinner

app = typer.Typer(invoke_without_command=True)


@app.callback(invoke_without_command=True)
def submit(
    ctx: typer.Context,
    description: str = typer.Argument(
        None,
        help="The work to be done (or use --file)",
    ),
    project: Optional[str] = typer.Option(
        None,
        "--project", "-p",
        help="Project ID",
    ),
    mode: str = typer.Option(
        "autonomous",
        "--mode", "-m",
        help="Mode: autonomous or collaborative",
    ),
    file: Optional[Path] = typer.Option(
        None,
        "--file", "-f",
        help="Read description from file",
    ),
    constitution: Optional[Path] = typer.Option(
        None,
        "--constitution", "-c",
        help="Path to constitution/standards file",
    ),
    priority: str = typer.Option(
        "normal",
        "--priority",
        help="Priority: low, normal, or high",
    ),
    tags: Optional[str] = typer.Option(
        None,
        "--tags", "-t",
        help="Comma-separated tags",
    ),
):
    """
    Submit a new request for processing.
    
    Examples:
        dats submit "Create a Python function that calculates fibonacci numbers"
        dats submit --project my-project "Add user authentication to the API"
        dats submit --mode collaborative "Let's design the database schema"
        dats submit --file request.md
    """
    formatter: Formatter = ctx.obj["formatter"]
    api_url: str = ctx.obj["api_url"]
    api_key: Optional[str] = ctx.obj["api_key"]
    
    # Get description from file or argument
    if file:
        if not file.exists():
            formatter.error(f"File not found: {file}")
            raise typer.Exit(1)
        description = file.read_text()
    
    if not description:
        formatter.error("Description required. Provide as argument or use --file")
        raise typer.Exit(1)
    
    # Build request
    payload = {
        "description": description,
        "mode": mode,
        "priority": priority,
    }
    
    if project:
        payload["project_id"] = project
    
    if constitution and constitution.exists():
        import yaml
        with open(constitution) as f:
            payload["constitution_override"] = yaml.safe_load(f)
    
    if tags:
        payload["metadata"] = {
            "tags": [t.strip() for t in tags.split(",")],
        }
    
    # Build headers
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["X-API-Key"] = api_key
    
    # Submit request
    formatter.debug(f"Connecting to API at {api_url}")
    formatter.debug(f"Request payload: {payload}")
    
    with get_progress_spinner() as progress:
        task = progress.add_task("Submitting request...", total=None)
        
        try:
            response = httpx.post(
                f"{api_url}/api/v1/tasks",
                json=payload,
                headers=headers,
                timeout=30.0,
            )
            
            if response.status_code == 201:
                data = response.json()
                formatter.print_task_submission(
                    task_id=data["task_id"],
                    status=data["status"],
                    mode=data["mode"],
                )
            else:
                error = response.json().get("error", {})
                formatter.error(
                    error.get("message", "Submission failed"),
                    code=error.get("code"),
                    details=error.get("details"),
                )
                raise typer.Exit(1)
                
        except httpx.RequestError as e:
            formatter.error(f"Connection error: {e}")
            raise typer.Exit(1)