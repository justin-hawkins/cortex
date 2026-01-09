"""
Config command for DATS CLI.

Handles CLI configuration management.
"""

from pathlib import Path
from typing import Optional

import typer
import yaml

from src.cli.output import Formatter

app = typer.Typer()

CONFIG_PATH = Path.home() / ".dats" / "config.yaml"


def _ensure_config_dir():
    """Ensure config directory exists."""
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)


def _load_config() -> dict:
    """Load configuration from file."""
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH) as f:
            return yaml.safe_load(f) or {}
    return {}


def _save_config(config: dict):
    """Save configuration to file."""
    _ensure_config_dir()
    with open(CONFIG_PATH, "w") as f:
        yaml.dump(config, f, default_flow_style=False)


@app.command("set")
def set_config(
    ctx: typer.Context,
    key: str = typer.Argument(
        ...,
        help="Config key (e.g., api_key, api_url)",
    ),
    value: str = typer.Argument(
        ...,
        help="Config value",
    ),
):
    """
    Set a configuration value.
    
    Examples:
        dats config set api_key <your-key>
        dats config set api_url http://localhost:8000
        dats config set default_project my-project
    """
    formatter: Formatter = ctx.obj["formatter"]
    
    config = _load_config()
    
    # Handle nested keys like api.url
    if "." in key:
        parts = key.split(".")
        current = config
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        current[parts[-1]] = value
    else:
        # Map simple keys to nested structure
        key_mapping = {
            "api_key": ("api", "key"),
            "api_url": ("api", "url"),
            "default_project": ("defaults", "project"),
            "default_mode": ("defaults", "mode"),
            "output_format": ("defaults", "output_format"),
        }
        
        if key in key_mapping:
            section, subkey = key_mapping[key]
            if section not in config:
                config[section] = {}
            config[section][subkey] = value
        else:
            config[key] = value
    
    _save_config(config)
    formatter.success(f"Set {key}={value}")


@app.command("get")
def get_config(
    ctx: typer.Context,
    key: Optional[str] = typer.Argument(
        None,
        help="Config key to get (shows all if not specified)",
    ),
):
    """
    Get configuration value(s).
    
    Examples:
        dats config get
        dats config get api_key
    """
    formatter: Formatter = ctx.obj["formatter"]
    
    config = _load_config()
    
    if key:
        # Handle nested keys
        if "." in key:
            parts = key.split(".")
            current = config
            for part in parts:
                if isinstance(current, dict) and part in current:
                    current = current[part]
                else:
                    formatter.error(f"Config key not found: {key}")
                    raise typer.Exit(1)
            formatter.console.print(f"{key}: {current}")
        else:
            # Map simple keys
            key_mapping = {
                "api_key": ("api", "key"),
                "api_url": ("api", "url"),
                "default_project": ("defaults", "project"),
                "default_mode": ("defaults", "mode"),
                "output_format": ("defaults", "output_format"),
            }
            
            if key in key_mapping:
                section, subkey = key_mapping[key]
                value = config.get(section, {}).get(subkey)
            else:
                value = config.get(key)
            
            if value is not None:
                formatter.console.print(f"{key}: {value}")
            else:
                formatter.error(f"Config key not found: {key}")
                raise typer.Exit(1)
    else:
        # Show all config
        if not config:
            formatter.console.print("[dim]No configuration set[/dim]")
            return
        
        formatter.console.print("\n[bold]Configuration:[/bold]")
        _print_config(formatter, config)


def _print_config(formatter: Formatter, config: dict, indent: int = 0):
    """Recursively print config."""
    prefix = "  " * indent
    for key, value in config.items():
        if isinstance(value, dict):
            formatter.console.print(f"{prefix}[cyan]{key}:[/cyan]")
            _print_config(formatter, value, indent + 1)
        else:
            # Mask sensitive values
            if key in ("key", "api_key", "password", "token"):
                display_value = value[:4] + "****" if value and len(value) > 4 else "****"
            else:
                display_value = value
            formatter.console.print(f"{prefix}[cyan]{key}:[/cyan] {display_value}")


@app.command("unset")
def unset_config(
    ctx: typer.Context,
    key: str = typer.Argument(
        ...,
        help="Config key to unset",
    ),
):
    """
    Unset a configuration value.
    
    Examples:
        dats config unset api_key
    """
    formatter: Formatter = ctx.obj["formatter"]
    
    config = _load_config()
    
    # Handle nested keys
    key_mapping = {
        "api_key": ("api", "key"),
        "api_url": ("api", "url"),
        "default_project": ("defaults", "project"),
        "default_mode": ("defaults", "mode"),
        "output_format": ("defaults", "output_format"),
    }
    
    if key in key_mapping:
        section, subkey = key_mapping[key]
        if section in config and subkey in config[section]:
            del config[section][subkey]
            if not config[section]:
                del config[section]
            _save_config(config)
            formatter.success(f"Unset {key}")
        else:
            formatter.error(f"Config key not found: {key}")
            raise typer.Exit(1)
    elif key in config:
        del config[key]
        _save_config(config)
        formatter.success(f"Unset {key}")
    else:
        formatter.error(f"Config key not found: {key}")
        raise typer.Exit(1)


@app.command("path")
def show_path(ctx: typer.Context):
    """
    Show the config file path.
    
    Examples:
        dats config path
    """
    formatter: Formatter = ctx.obj["formatter"]
    formatter.console.print(f"Config file: {CONFIG_PATH}")


@app.command("edit")
def edit_config(ctx: typer.Context):
    """
    Open config file in editor.
    
    Examples:
        dats config edit
    """
    import os
    import subprocess
    
    formatter: Formatter = ctx.obj["formatter"]
    
    _ensure_config_dir()
    
    # Create default config if it doesn't exist
    if not CONFIG_PATH.exists():
        default_config = {
            "api": {
                "url": "http://localhost:8000",
            },
            "defaults": {
                "project": "default",
                "mode": "autonomous",
                "output_format": "human",
            },
            "display": {
                "color": True,
                "verbose": False,
            },
        }
        _save_config(default_config)
    
    # Open in editor
    editor = os.environ.get("EDITOR", "vim")
    try:
        subprocess.run([editor, str(CONFIG_PATH)], check=True)
        formatter.success("Configuration updated")
    except subprocess.CalledProcessError:
        formatter.error("Editor exited with error")
        raise typer.Exit(1)
    except FileNotFoundError:
        formatter.error(f"Editor '{editor}' not found. Set EDITOR environment variable.")
        raise typer.Exit(1)