"""
Project command for DATS CLI.

Handles project management operations.
"""

from pathlib import Path
from typing import Optional

import httpx
import typer

from src.cli.output import Formatter

app = typer.Typer()


@app.command("init")
def init_project(
    ctx: typer.Context,
    name: str = typer.Argument(
        ...,
        help="Project name",
    ),
    description: Optional[str] = typer.Option(
        None,
        "--description", "-d",
        help="Project description",
    ),
    qa_profile: str = typer.Option(
        "consensus",
        "--qa-profile",
        help="QA profile (consensus, adversarial, security)",
    ),
):
    """
    Initialize a new project.
    
    Examples:
        dats project init my-project
        dats project init my-project --description "My awesome project"
    """
    formatter: Formatter = ctx.obj["formatter"]
    api_url: str = ctx.obj["api_url"]
    api_key: Optional[str] = ctx.obj["api_key"]
    
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["X-API-Key"] = api_key
    
    payload = {
        "name": name,
        "description": description,
        "config": {
            "qa_profile": qa_profile,
            "default_mode": "autonomous",
        },
    }
    
    try:
        response = httpx.post(
            f"{api_url}/api/v1/projects",
            headers=headers,
            json=payload,
            timeout=30.0,
        )
        
        if response.status_code == 201:
            data = response.json()
            formatter.success(f"Project '{name}' created", data={
                "ID": data.get("id"),
                "QA Profile": qa_profile,
            })
        elif response.status_code == 409:
            formatter.error(f"Project '{name}' already exists")
            raise typer.Exit(1)
        else:
            error = response.json().get("error", {})
            formatter.error(
                error.get("message", "Failed to create project"),
                code=error.get("code"),
            )
            raise typer.Exit(1)
            
    except httpx.RequestError as e:
        formatter.error(f"Connection error: {e}")
        raise typer.Exit(1)


@app.command("list")
def list_projects(
    ctx: typer.Context,
):
    """
    List all projects.
    
    Examples:
        dats project list
    """
    formatter: Formatter = ctx.obj["formatter"]
    api_url: str = ctx.obj["api_url"]
    api_key: Optional[str] = ctx.obj["api_key"]
    
    headers = {}
    if api_key:
        headers["X-API-Key"] = api_key
    
    try:
        response = httpx.get(
            f"{api_url}/api/v1/projects",
            headers=headers,
            timeout=30.0,
        )
        
        if response.status_code == 200:
            data = response.json()
            formatter.print_project_list(data.get("projects", []))
        else:
            error = response.json().get("error", {})
            formatter.error(
                error.get("message", "Failed to list projects"),
                code=error.get("code"),
            )
            raise typer.Exit(1)
            
    except httpx.RequestError as e:
        formatter.error(f"Connection error: {e}")
        raise typer.Exit(1)


@app.command("config")
def configure_project(
    ctx: typer.Context,
    project_id: str = typer.Argument(
        ...,
        help="Project ID",
    ),
    set_config: Optional[str] = typer.Option(
        None,
        "--set",
        help="Set config value (key=value)",
    ),
):
    """
    View or set project configuration.
    
    Examples:
        dats project config my-project
        dats project config my-project --set qa_profile=adversarial
    """
    formatter: Formatter = ctx.obj["formatter"]
    api_url: str = ctx.obj["api_url"]
    api_key: Optional[str] = ctx.obj["api_key"]
    
    headers = {}
    if api_key:
        headers["X-API-Key"] = api_key
    
    if set_config:
        # Update config
        try:
            key, value = set_config.split("=", 1)
        except ValueError:
            formatter.error("Invalid format. Use: key=value")
            raise typer.Exit(1)
        
        headers["Content-Type"] = "application/json"
        
        try:
            # Get current config first
            response = httpx.get(
                f"{api_url}/api/v1/projects/{project_id}",
                headers=headers,
                timeout=30.0,
            )
            
            if response.status_code != 200:
                formatter.error(f"Project not found: {project_id}")
                raise typer.Exit(1)
            
            current = response.json()
            current_config = current.get("config", {})
            current_config[key] = value
            
            # Update
            response = httpx.put(
                f"{api_url}/api/v1/projects/{project_id}",
                headers=headers,
                json={"config": current_config},
                timeout=30.0,
            )
            
            if response.status_code == 200:
                formatter.success(f"Set {key}={value}")
            else:
                error = response.json().get("error", {})
                formatter.error(
                    error.get("message", "Failed to update config"),
                    code=error.get("code"),
                )
                raise typer.Exit(1)
                
        except httpx.RequestError as e:
            formatter.error(f"Connection error: {e}")
            raise typer.Exit(1)
    else:
        # Show config
        try:
            response = httpx.get(
                f"{api_url}/api/v1/projects/{project_id}",
                headers=headers,
                timeout=30.0,
            )
            
            if response.status_code == 200:
                data = response.json()
                formatter.console.print(f"\nProject: [cyan]{project_id}[/cyan]")
                formatter.console.print(f"Name: {data.get('name', 'N/A')}")
                formatter.console.print(f"Description: {data.get('description', 'N/A')}")
                formatter.console.print(f"Tasks: {data.get('task_count', 0)}")
                formatter.console.print("\nConfiguration:")
                config = data.get("config", {})
                for key, value in config.items():
                    formatter.console.print(f"  {key}: {value}")
            elif response.status_code == 404:
                formatter.error(f"Project not found: {project_id}")
                raise typer.Exit(1)
            else:
                error = response.json().get("error", {})
                formatter.error(
                    error.get("message", "Failed to get project"),
                    code=error.get("code"),
                )
                raise typer.Exit(1)
                
        except httpx.RequestError as e:
            formatter.error(f"Connection error: {e}")
            raise typer.Exit(1)


@app.command("link")
def link_repo(
    ctx: typer.Context,
    project_id: str = typer.Argument(
        ...,
        help="Project ID",
    ),
    repo: str = typer.Option(
        ...,
        "--repo", "-r",
        help="Git repository URL",
    ),
):
    """
    Link a git repository to a project.
    
    Examples:
        dats project link my-project --repo git@github.com:user/repo.git
    """
    formatter: Formatter = ctx.obj["formatter"]
    api_url: str = ctx.obj["api_url"]
    api_key: Optional[str] = ctx.obj["api_key"]
    
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["X-API-Key"] = api_key
    
    try:
        response = httpx.put(
            f"{api_url}/api/v1/projects/{project_id}",
            headers=headers,
            json={"repo_url": repo},
            timeout=30.0,
        )
        
        if response.status_code == 200:
            formatter.success(f"Linked repository to project {project_id}")
        elif response.status_code == 404:
            formatter.error(f"Project not found: {project_id}")
            raise typer.Exit(1)
        else:
            error = response.json().get("error", {})
            formatter.error(
                error.get("message", "Failed to link repository"),
                code=error.get("code"),
            )
            raise typer.Exit(1)
            
    except httpx.RequestError as e:
        formatter.error(f"Connection error: {e}")
        raise typer.Exit(1)


@app.command("delete")
def delete_project(
    ctx: typer.Context,
    project_id: str = typer.Argument(
        ...,
        help="Project ID",
    ),
    force: bool = typer.Option(
        False,
        "--force", "-f",
        help="Skip confirmation",
    ),
):
    """
    Delete a project.
    
    Examples:
        dats project delete my-project
        dats project delete my-project --force
    """
    formatter: Formatter = ctx.obj["formatter"]
    api_url: str = ctx.obj["api_url"]
    api_key: Optional[str] = ctx.obj["api_key"]
    
    if not force:
        confirm = typer.confirm(f"Are you sure you want to delete project '{project_id}'?")
        if not confirm:
            formatter.info("Cancelled")
            raise typer.Exit(0)
    
    headers = {}
    if api_key:
        headers["X-API-Key"] = api_key
    
    try:
        response = httpx.delete(
            f"{api_url}/api/v1/projects/{project_id}",
            headers=headers,
            timeout=30.0,
        )
        
        if response.status_code == 204:
            formatter.success(f"Project '{project_id}' deleted")
        elif response.status_code == 404:
            formatter.error(f"Project not found: {project_id}")
            raise typer.Exit(1)
        elif response.status_code == 400:
            error = response.json().get("error", {})
            formatter.error(
                error.get("message", "Cannot delete project"),
                code=error.get("code"),
            )
            raise typer.Exit(1)
        else:
            error = response.json().get("error", {})
            formatter.error(
                error.get("message", "Failed to delete project"),
                code=error.get("code"),
            )
            raise typer.Exit(1)
            
    except httpx.RequestError as e:
        formatter.error(f"Connection error: {e}")
        raise typer.Exit(1)