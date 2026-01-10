"""
DATS CLI main entry point.

The main Typer application that provides all CLI commands.
"""

import os
from pathlib import Path
from typing import Optional

import typer

from src.cli.output import Formatter, OutputFormat

# Create main app
app = typer.Typer(
    name="dats",
    help="Distributed Agentic Task System CLI",
    no_args_is_help=True,
)

# Import and include command groups
from src.cli.commands import config, monitoring, project, review, status, submit

app.add_typer(submit.app, name="submit", help="Submit work")
app.add_typer(status.app, name="status", help="Check status")
app.add_typer(review.app, name="review", help="Human review actions")
app.add_typer(project.app, name="project", help="Project management")
app.add_typer(config.app, name="config", help="Configuration management")

# Add monitoring commands directly to main app
app.command("workers")(monitoring.workers)
app.command("queues")(monitoring.queues)
app.command("failures")(monitoring.failures)
app.command("provenance")(monitoring.provenance_export)


# Global options
@app.callback()
def main(
    ctx: typer.Context,
    output: str = typer.Option(
        "human",
        "--output", "-o",
        help="Output format (human, json)",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose", "-v",
        help="Enable verbose output",
    ),
    api_url: Optional[str] = typer.Option(
        None,
        "--api-url",
        envvar="DATS_API_URL",
        help="API URL",
    ),
    api_key: Optional[str] = typer.Option(
        None,
        "--api-key",
        envvar="DATS_API_KEY",
        help="API key",
    ),
):
    """
    DATS - Distributed Agentic Task System CLI.
    
    Submit and manage work in the distributed agentic task system.
    """
    # Initialize context
    ctx.ensure_object(dict)
    
    # Set output format
    try:
        output_format = OutputFormat(output.lower())
    except ValueError:
        output_format = OutputFormat.HUMAN
    
    ctx.obj["formatter"] = Formatter(
        format=output_format,
        verbose=verbose,
    )
    ctx.obj["api_url"] = api_url or _get_api_url()
    ctx.obj["api_key"] = api_key or _get_api_key()
    ctx.obj["verbose"] = verbose


def _get_api_url() -> str:
    """Get API URL from config or default."""
    # Check config file
    config_path = Path.home() / ".dats" / "config.yaml"
    if config_path.exists():
        try:
            import yaml
            with open(config_path) as f:
                cfg = yaml.safe_load(f)
                if cfg and cfg.get("api", {}).get("url"):
                    return cfg["api"]["url"]
        except Exception:
            pass
    
    # Fall back to default
    return "http://localhost:8000"


def _get_api_key() -> Optional[str]:
    """Get API key from config."""
    # Check config file
    config_path = Path.home() / ".dats" / "config.yaml"
    if config_path.exists():
        try:
            import yaml
            with open(config_path) as f:
                cfg = yaml.safe_load(f)
                if cfg and cfg.get("api", {}).get("key"):
                    return cfg["api"]["key"]
        except Exception:
            pass
    
    return None


@app.command()
def version():
    """Show DATS version."""
    from src.api.app import API_VERSION
    typer.echo(f"DATS CLI v{API_VERSION}")


if __name__ == "__main__":
    app()