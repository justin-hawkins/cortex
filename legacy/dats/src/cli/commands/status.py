"""
Status command for DATS CLI.

Handles checking task status and progress.
"""

import asyncio
import time
from typing import Optional

import httpx
import typer

from src.cli.output import Formatter, get_progress_spinner

app = typer.Typer(invoke_without_command=True)


@app.callback(invoke_without_command=True)
def status(
    ctx: typer.Context,
    task_id: str = typer.Argument(
        None,
        help="Task ID to check status for",
    ),
    project: Optional[str] = typer.Option(
        None,
        "--project", "-p",
        help="Get status of all tasks in project",
    ),
    watch: bool = typer.Option(
        False,
        "--watch", "-w",
        help="Watch task progress (live updates)",
    ),
    summary: bool = typer.Option(
        False,
        "--summary", "-s",
        help="Show project summary",
    ),
    tree: bool = typer.Option(
        False,
        "--tree", "-t",
        help="Show task decomposition tree",
    ),
    wait: bool = typer.Option(
        False,
        "--wait",
        help="Wait for task completion",
    ),
    timeout: int = typer.Option(
        3600,
        "--timeout",
        help="Timeout in seconds for --wait",
    ),
):
    """
    Check status of tasks.
    
    Examples:
        dats status <task-id>
        dats status --project my-project
        dats status --watch <task-id>
        dats status --project my-project --summary
    """
    formatter: Formatter = ctx.obj["formatter"]
    api_url: str = ctx.obj["api_url"]
    api_key: Optional[str] = ctx.obj["api_key"]
    
    # Build headers
    headers = {}
    if api_key:
        headers["X-API-Key"] = api_key
    
    if project and not task_id:
        # List tasks in project
        _list_project_tasks(formatter, api_url, headers, project, summary)
    elif task_id:
        if tree:
            _show_task_tree(formatter, api_url, headers, task_id)
        elif watch:
            _watch_task(formatter, api_url, headers, task_id)
        elif wait:
            _wait_for_task(formatter, api_url, headers, task_id, timeout)
        else:
            _show_task_status(formatter, api_url, headers, task_id)
    else:
        formatter.error("Please provide a task ID or use --project")
        raise typer.Exit(1)


def _show_task_status(
    formatter: Formatter,
    api_url: str,
    headers: dict,
    task_id: str,
):
    """Show status of a single task."""
    try:
        response = httpx.get(
            f"{api_url}/api/v1/tasks/{task_id}",
            headers=headers,
            timeout=30.0,
        )
        
        if response.status_code == 200:
            formatter.print_task_status(response.json())
        elif response.status_code == 404:
            formatter.error(f"Task not found: {task_id}")
            raise typer.Exit(1)
        else:
            error = response.json().get("error", {})
            formatter.error(
                error.get("message", "Failed to get status"),
                code=error.get("code"),
            )
            raise typer.Exit(1)
            
    except httpx.RequestError as e:
        formatter.error(f"Connection error: {e}")
        raise typer.Exit(1)


def _show_task_tree(
    formatter: Formatter,
    api_url: str,
    headers: dict,
    task_id: str,
):
    """Show task decomposition tree."""
    try:
        response = httpx.get(
            f"{api_url}/api/v1/tasks/{task_id}/tree",
            headers=headers,
            timeout=30.0,
        )
        
        if response.status_code == 200:
            data = response.json()
            formatter.print_task_tree(data.get("root", {}))
        elif response.status_code == 404:
            formatter.error(f"Task tree not found: {task_id}")
            raise typer.Exit(1)
        else:
            error = response.json().get("error", {})
            formatter.error(
                error.get("message", "Failed to get tree"),
                code=error.get("code"),
            )
            raise typer.Exit(1)
            
    except httpx.RequestError as e:
        formatter.error(f"Connection error: {e}")
        raise typer.Exit(1)


def _list_project_tasks(
    formatter: Formatter,
    api_url: str,
    headers: dict,
    project: str,
    summary: bool,
):
    """List tasks in a project."""
    try:
        response = httpx.get(
            f"{api_url}/api/v1/projects/{project}/tasks",
            headers=headers,
            timeout=30.0,
        )
        
        if response.status_code == 200:
            data = response.json()
            
            if summary:
                # Print summary statistics
                tasks = data.get("tasks", [])
                total = data.get("total", len(tasks))
                
                completed = sum(1 for t in tasks if t.get("status") == "completed")
                failed = sum(1 for t in tasks if t.get("status") == "failed")
                in_progress = sum(1 for t in tasks if t.get("status") in ("in_progress", "decomposing"))
                queued = sum(1 for t in tasks if t.get("status") == "queued")
                
                formatter.console.print(f"\nProject: [cyan]{project}[/cyan]")
                formatter.console.print(f"Total tasks: {total}")
                formatter.console.print(f"  [green]Completed: {completed}[/green]")
                formatter.console.print(f"  [cyan]In Progress: {in_progress}[/cyan]")
                formatter.console.print(f"  [yellow]Queued: {queued}[/yellow]")
                formatter.console.print(f"  [red]Failed: {failed}[/red]")
            else:
                formatter.print_task_list(data.get("tasks", []), data.get("total", 0))
        elif response.status_code == 404:
            formatter.error(f"Project not found: {project}")
            raise typer.Exit(1)
        else:
            error = response.json().get("error", {})
            formatter.error(
                error.get("message", "Failed to list tasks"),
                code=error.get("code"),
            )
            raise typer.Exit(1)
            
    except httpx.RequestError as e:
        formatter.error(f"Connection error: {e}")
        raise typer.Exit(1)


def _watch_task(
    formatter: Formatter,
    api_url: str,
    headers: dict,
    task_id: str,
):
    """Watch task progress with live updates."""
    import json
    
    formatter.info(f"Watching task {task_id} (Ctrl+C to stop)")
    
    try:
        with httpx.stream(
            "GET",
            f"{api_url}/api/v1/tasks/{task_id}/stream",
            headers=headers,
            timeout=None,
        ) as response:
            for line in response.iter_lines():
                if line.startswith("data:"):
                    try:
                        data = json.loads(line[5:].strip())
                        event_type = data.get("type", "unknown")
                        
                        if event_type == "status_update":
                            status = data.get("status", "unknown")
                            progress = data.get("progress", {})
                            formatter.console.print(
                                f"[{_status_color(status)}]●[/{_status_color(status)}] "
                                f"Status: {status} "
                                f"({progress.get('completed', 0)}/{progress.get('total', 0)} subtasks)"
                            )
                        elif event_type == "task_complete":
                            formatter.console.print("[green]✓ Task completed![/green]")
                            break
                        elif event_type == "task_failed":
                            formatter.console.print(f"[red]✗ Task failed: {data.get('error', 'Unknown')}[/red]")
                            break
                        elif event_type == "review_needed":
                            formatter.console.print(
                                f"[yellow]! Human review needed: {data.get('review_id')}[/yellow]"
                            )
                    except json.JSONDecodeError:
                        pass
                        
    except KeyboardInterrupt:
        formatter.info("Stopped watching")
    except httpx.RequestError as e:
        formatter.error(f"Connection error: {e}")
        raise typer.Exit(1)


def _wait_for_task(
    formatter: Formatter,
    api_url: str,
    headers: dict,
    task_id: str,
    timeout: int,
):
    """Wait for task completion."""
    formatter.info(f"Waiting for task {task_id} to complete (timeout: {timeout}s)")
    
    start_time = time.time()
    
    with get_progress_spinner() as progress:
        task = progress.add_task("Waiting for completion...", total=None)
        
        while time.time() - start_time < timeout:
            try:
                response = httpx.get(
                    f"{api_url}/api/v1/tasks/{task_id}",
                    headers=headers,
                    timeout=30.0,
                )
                
                if response.status_code == 200:
                    data = response.json()
                    status = data.get("status", "unknown")
                    
                    if status == "completed":
                        formatter.success("Task completed!")
                        formatter.print_task_status(data)
                        return
                    elif status == "failed":
                        formatter.error("Task failed!")
                        formatter.print_task_status(data)
                        raise typer.Exit(1)
                    
                    # Update progress
                    progress.update(
                        task,
                        description=f"Status: {status}...",
                    )
                    
            except httpx.RequestError:
                pass
            
            time.sleep(5)
        
        formatter.error("Timeout waiting for task completion")
        raise typer.Exit(1)


def _status_color(status: str) -> str:
    """Get color for status."""
    return {
        "queued": "yellow",
        "decomposing": "blue",
        "in_progress": "cyan",
        "validating": "magenta",
        "completed": "green",
        "failed": "red",
        "cancelled": "dim",
    }.get(status, "white")