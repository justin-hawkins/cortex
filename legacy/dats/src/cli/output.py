"""
CLI output formatting helpers.

Provides consistent formatting for human-readable and JSON output.
"""

import json
import sys
from datetime import datetime
from enum import Enum
from typing import Any, Optional

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.tree import Tree


class OutputFormat(str, Enum):
    """Output format options."""

    HUMAN = "human"
    JSON = "json"


# Global console for output
console = Console()
error_console = Console(stderr=True)


class Formatter:
    """Output formatter with support for multiple formats."""

    def __init__(
        self,
        format: OutputFormat = OutputFormat.HUMAN,
        verbose: bool = False,
        color: bool = True,
    ):
        """
        Initialize formatter.
        
        Args:
            format: Output format (human or json)
            verbose: Enable verbose output
            color: Enable colored output
        """
        self.format = format
        self.verbose = verbose
        self.color = color
        self.console = Console(force_terminal=color, no_color=not color)

    def success(self, message: str, data: Optional[dict] = None):
        """Display success message."""
        if self.format == OutputFormat.JSON:
            output = {"status": "success", "message": message}
            if data:
                output.update(data)
            print(json.dumps(output, indent=2, default=str))
        else:
            self.console.print(f"[green]✓[/green] {message}")
            if data and self.verbose:
                for key, value in data.items():
                    self.console.print(f"  [dim]{key}:[/dim] {value}")

    def error(self, message: str, code: Optional[str] = None, details: Optional[dict] = None):
        """Display error message."""
        if self.format == OutputFormat.JSON:
            output = {
                "status": "error",
                "error": {
                    "message": message,
                    "code": code or "ERROR",
                    "details": details,
                },
            }
            print(json.dumps(output, indent=2, default=str))
        else:
            error_console.print(f"[red]✗[/red] {message}")
            if code:
                error_console.print(f"  [dim]Code:[/dim] {code}")
            if details:
                for key, value in details.items():
                    error_console.print(f"  [dim]{key}:[/dim] {value}")

    def warning(self, message: str):
        """Display warning message."""
        if self.format == OutputFormat.JSON:
            print(json.dumps({"status": "warning", "message": message}, default=str))
        else:
            self.console.print(f"[yellow]![/yellow] {message}")

    def info(self, message: str):
        """Display info message."""
        if self.format == OutputFormat.JSON:
            pass  # Info messages not included in JSON
        else:
            self.console.print(f"[blue]ℹ[/blue] {message}")

    def debug(self, message: str):
        """Display debug message (only in verbose mode)."""
        if self.verbose and self.format != OutputFormat.JSON:
            self.console.print(f"[dim][DEBUG][/dim] {message}")

    def print_json(self, data: Any):
        """Print data as JSON."""
        print(json.dumps(data, indent=2, default=str))

    def print_task_submission(self, task_id: str, status: str, mode: str):
        """Print task submission result."""
        if self.format == OutputFormat.JSON:
            self.print_json({
                "task_id": task_id,
                "status": status,
                "mode": mode,
            })
        else:
            self.console.print()
            self.console.print(Panel.fit(
                f"[green]✓[/green] Request submitted\n"
                f"  [dim]Task ID:[/dim] [cyan]{task_id}[/cyan]\n"
                f"  [dim]Mode:[/dim] {mode}\n"
                f"  [dim]Status:[/dim] {status}\n"
                f"\n"
                f"  [dim]Track progress:[/dim] dats status --watch {task_id}",
                title="DATS",
            ))

    def print_task_status(self, task: dict):
        """Print task status."""
        if self.format == OutputFormat.JSON:
            self.print_json(task)
        else:
            status = task.get("status", "unknown")
            status_color = {
                "queued": "yellow",
                "decomposing": "blue",
                "in_progress": "cyan",
                "validating": "magenta",
                "completed": "green",
                "failed": "red",
                "cancelled": "dim",
            }.get(status, "white")

            self.console.print()
            self.console.print(f"Task: [cyan]{task.get('task_id', 'unknown')}[/cyan]")
            self.console.print(f"Status: [{status_color}]{status}[/{status_color}]")
            self.console.print(f"Mode: {task.get('mode', 'unknown')}")

            # Progress
            progress = task.get("progress", {})
            if progress:
                total = progress.get("total_subtasks", 0)
                completed = progress.get("completed_subtasks", 0)
                failed = progress.get("failed_subtasks", 0)
                
                if total > 0:
                    self.console.print()
                    self.console.print(f"Progress: {completed}/{total} subtasks")
                    if failed > 0:
                        self.console.print(f"  [red]Failed: {failed}[/red]")

            # Current activity
            activity = task.get("current_activity")
            if activity:
                self.console.print()
                self.console.print(f"Current: {activity.get('description', 'Unknown')}")

            # Human actions needed
            actions = task.get("human_actions_needed", [])
            if actions:
                self.console.print()
                self.console.print("[yellow]Human action required:[/yellow]")
                for action in actions:
                    self.console.print(f"  • {action.get('summary', 'Review needed')}")
                    self.console.print(f"    [dim]Review ID:[/dim] {action.get('review_id')}")

    def print_task_list(self, tasks: list[dict], total: int):
        """Print list of tasks."""
        if self.format == OutputFormat.JSON:
            self.print_json({"tasks": tasks, "total": total})
        else:
            if not tasks:
                self.console.print("[dim]No tasks found[/dim]")
                return

            table = Table(title="Tasks")
            table.add_column("ID", style="cyan")
            table.add_column("Status")
            table.add_column("Mode")
            table.add_column("Description")
            table.add_column("Created")

            for task in tasks:
                status = task.get("status", "unknown")
                status_color = {
                    "queued": "yellow",
                    "in_progress": "cyan",
                    "completed": "green",
                    "failed": "red",
                }.get(status, "white")

                desc = task.get("description", "")[:50]
                if len(task.get("description", "")) > 50:
                    desc += "..."

                created = task.get("created_at", "")
                if isinstance(created, datetime):
                    created = created.strftime("%Y-%m-%d %H:%M")

                table.add_row(
                    task.get("task_id", "")[:12] + "...",
                    f"[{status_color}]{status}[/{status_color}]",
                    task.get("mode", ""),
                    desc,
                    str(created),
                )

            self.console.print(table)
            self.console.print(f"\n[dim]Total: {total}[/dim]")

    def print_review_list(self, reviews: list[dict], total: int):
        """Print list of pending reviews."""
        if self.format == OutputFormat.JSON:
            self.print_json({"reviews": reviews, "total": total})
        else:
            if not reviews:
                self.console.print("[dim]No pending reviews[/dim]")
                return

            table = Table(title="Pending Reviews")
            table.add_column("ID", style="cyan")
            table.add_column("Type")
            table.add_column("Task ID")
            table.add_column("Summary")
            table.add_column("Created")

            for review in reviews:
                created = review.get("created_at", "")
                if isinstance(created, datetime):
                    created = created.strftime("%Y-%m-%d %H:%M")

                summary = review.get("summary", "")[:40]
                if len(review.get("summary", "")) > 40:
                    summary += "..."

                table.add_row(
                    review.get("review_id", "")[:12] + "...",
                    review.get("type", ""),
                    review.get("task_id", "")[:12] + "...",
                    summary,
                    str(created),
                )

            self.console.print(table)
            self.console.print(f"\n[dim]Total: {total}[/dim]")

    def print_review_detail(self, review: dict):
        """Print review details."""
        if self.format == OutputFormat.JSON:
            self.print_json(review)
        else:
            self.console.print()
            self.console.print(Panel.fit(
                f"[cyan]Review ID:[/cyan] {review.get('review_id', 'unknown')}\n"
                f"[dim]Type:[/dim] {review.get('type', 'unknown')}\n"
                f"[dim]Task ID:[/dim] {review.get('task_id', 'unknown')}\n"
                f"[dim]Status:[/dim] {review.get('status', 'unknown')}",
                title="Review Details",
            ))

            context = review.get("context", {})
            if context:
                self.console.print()
                self.console.print("[bold]Context:[/bold]")
                self.console.print(f"  Task: {context.get('task_description', 'N/A')}")
                self.console.print(f"  Output: {context.get('output_summary', 'N/A')[:200]}")

                questions = context.get("specific_questions", [])
                if questions:
                    self.console.print()
                    self.console.print("[bold]Questions for reviewer:[/bold]")
                    for q in questions:
                        self.console.print(f"  • {q}")

            options = review.get("options", [])
            if options:
                self.console.print()
                self.console.print("[bold]Decision options:[/bold]")
                for opt in options:
                    self.console.print(f"  [{opt.get('id')}] {opt.get('description')}")

            recommendation = review.get("recommendation")
            if recommendation:
                self.console.print()
                self.console.print(f"[green]Recommendation:[/green] {recommendation}")

    def print_project_list(self, projects: list[dict]):
        """Print list of projects."""
        if self.format == OutputFormat.JSON:
            self.print_json({"projects": projects})
        else:
            if not projects:
                self.console.print("[dim]No projects found[/dim]")
                return

            table = Table(title="Projects")
            table.add_column("ID", style="cyan")
            table.add_column("Name")
            table.add_column("Tasks")
            table.add_column("QA Profile")
            table.add_column("Description")

            for project in projects:
                desc = project.get("description", "") or ""
                if len(desc) > 30:
                    desc = desc[:30] + "..."

                config = project.get("config", {})
                
                table.add_row(
                    project.get("id", ""),
                    project.get("name", ""),
                    str(project.get("task_count", 0)),
                    config.get("qa_profile", "default"),
                    desc,
                )

            self.console.print(table)

    def print_workers(self, workers: list[dict]):
        """Print worker status."""
        if self.format == OutputFormat.JSON:
            self.print_json({"workers": workers})
        else:
            if not workers:
                self.console.print("[dim]No workers connected[/dim]")
                return

            table = Table(title="Workers")
            table.add_column("Worker ID", style="cyan")
            table.add_column("Status")
            table.add_column("Tier")
            table.add_column("Current Task")
            table.add_column("Completed")

            for worker in workers:
                status = worker.get("status", "unknown")
                status_color = "green" if status == "active" else "yellow"

                task = worker.get("current_task")
                task_display = task[:12] + "..." if task else "-"

                table.add_row(
                    worker.get("worker_id", ""),
                    f"[{status_color}]{status}[/{status_color}]",
                    worker.get("tier", "unknown"),
                    task_display,
                    str(worker.get("tasks_completed", 0)),
                )

            self.console.print(table)

    def print_queues(self, queues: list[dict]):
        """Print queue status."""
        if self.format == OutputFormat.JSON:
            self.print_json({"queues": queues})
        else:
            table = Table(title="Queues")
            table.add_column("Queue", style="cyan")
            table.add_column("Depth")
            table.add_column("Consumers")

            for queue in queues:
                depth = queue.get("depth", 0)
                depth_color = "green" if depth < 10 else "yellow" if depth < 50 else "red"

                table.add_row(
                    queue.get("name", ""),
                    f"[{depth_color}]{depth}[/{depth_color}]",
                    str(queue.get("consumers", 0)),
                )

            self.console.print(table)

    def print_failures(self, failures: list[dict]):
        """Print recent failures."""
        if self.format == OutputFormat.JSON:
            self.print_json({"failures": failures})
        else:
            if not failures:
                self.console.print("[green]No recent failures[/green]")
                return

            table = Table(title="Recent Failures")
            table.add_column("Task ID", style="cyan")
            table.add_column("Tier")
            table.add_column("Error")
            table.add_column("Failed At")

            for failure in failures:
                failed_at = failure.get("failed_at", "")
                if isinstance(failed_at, datetime):
                    failed_at = failed_at.strftime("%Y-%m-%d %H:%M")

                error = failure.get("error", "Unknown")[:50]

                table.add_row(
                    failure.get("task_id", "")[:12] + "...",
                    failure.get("tier", "unknown"),
                    error,
                    str(failed_at),
                )

            self.console.print(table)

    def print_task_tree(self, root: dict, level: int = 0):
        """Print task decomposition tree."""
        if self.format == OutputFormat.JSON:
            self.print_json(root)
        else:
            tree = Tree(f"[cyan]{root.get('task_id', 'Root')}[/cyan]")
            self._add_tree_nodes(tree, root)
            self.console.print(tree)

    def _add_tree_nodes(self, tree: Tree, node: dict):
        """Recursively add nodes to tree."""
        status = node.get("status", "unknown")
        status_color = {
            "completed": "green",
            "failed": "red",
            "in_progress": "cyan",
        }.get(status, "dim")

        desc = node.get("description", "")[:50]
        tier = node.get("tier", "")
        tier_str = f"[dim]({tier})[/dim]" if tier else ""

        for child in node.get("children", []):
            child_status = child.get("status", "unknown")
            child_color = {
                "completed": "green",
                "failed": "red",
                "in_progress": "cyan",
            }.get(child_status, "dim")

            child_desc = child.get("description", "")[:40]
            child_tier = child.get("tier", "")
            child_tier_str = f"[dim]({child_tier})[/dim]" if child_tier else ""

            branch = tree.add(
                f"[{child_color}]●[/{child_color}] {child_desc} {child_tier_str}"
            )
            self._add_tree_nodes(branch, child)


def get_progress_spinner() -> Progress:
    """Get a progress spinner for long operations."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    )