"""
Review command for DATS CLI.

Handles human review operations.
"""

from typing import Optional

import httpx
import typer

from src.cli.output import Formatter

app = typer.Typer()


@app.command("list")
def list_reviews(
    ctx: typer.Context,
    project: Optional[str] = typer.Option(
        None,
        "--project", "-p",
        help="Filter by project",
    ),
    priority: Optional[str] = typer.Option(
        None,
        "--priority",
        help="Filter by priority (low, normal, high, critical)",
    ),
):
    """
    List pending reviews.
    
    Examples:
        dats review list
        dats review list --project my-project
    """
    formatter: Formatter = ctx.obj["formatter"]
    api_url: str = ctx.obj["api_url"]
    api_key: Optional[str] = ctx.obj["api_key"]
    
    headers = {}
    if api_key:
        headers["X-API-Key"] = api_key
    
    params = {}
    if project:
        params["project_id"] = project
    if priority:
        params["priority"] = priority
    
    try:
        response = httpx.get(
            f"{api_url}/api/v1/reviews",
            headers=headers,
            params=params,
            timeout=30.0,
        )
        
        if response.status_code == 200:
            data = response.json()
            formatter.print_review_list(data.get("reviews", []), data.get("total", 0))
        else:
            error = response.json().get("error", {})
            formatter.error(
                error.get("message", "Failed to list reviews"),
                code=error.get("code"),
            )
            raise typer.Exit(1)
            
    except httpx.RequestError as e:
        formatter.error(f"Connection error: {e}")
        raise typer.Exit(1)


@app.command("show")
def show_review(
    ctx: typer.Context,
    review_id: str = typer.Argument(
        ...,
        help="Review ID",
    ),
):
    """
    View review details.
    
    Examples:
        dats review show <review-id>
    """
    formatter: Formatter = ctx.obj["formatter"]
    api_url: str = ctx.obj["api_url"]
    api_key: Optional[str] = ctx.obj["api_key"]
    
    headers = {}
    if api_key:
        headers["X-API-Key"] = api_key
    
    try:
        response = httpx.get(
            f"{api_url}/api/v1/reviews/{review_id}",
            headers=headers,
            timeout=30.0,
        )
        
        if response.status_code == 200:
            formatter.print_review_detail(response.json())
        elif response.status_code == 404:
            formatter.error(f"Review not found: {review_id}")
            raise typer.Exit(1)
        else:
            error = response.json().get("error", {})
            formatter.error(
                error.get("message", "Failed to get review"),
                code=error.get("code"),
            )
            raise typer.Exit(1)
            
    except httpx.RequestError as e:
        formatter.error(f"Connection error: {e}")
        raise typer.Exit(1)


@app.command("approve")
def approve_review(
    ctx: typer.Context,
    review_id: str = typer.Argument(
        ...,
        help="Review ID",
    ),
    comments: Optional[str] = typer.Option(
        None,
        "--comments", "-c",
        help="Optional comments",
    ),
):
    """
    Approve a review.
    
    Examples:
        dats review approve <review-id>
        dats review approve <review-id> --comments "Looks good"
    """
    formatter: Formatter = ctx.obj["formatter"]
    api_url: str = ctx.obj["api_url"]
    api_key: Optional[str] = ctx.obj["api_key"]
    
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["X-API-Key"] = api_key
    
    payload = {}
    if comments:
        payload["comments"] = comments
    
    try:
        response = httpx.post(
            f"{api_url}/api/v1/reviews/{review_id}/approve",
            headers=headers,
            json=payload,
            timeout=30.0,
        )
        
        if response.status_code == 200:
            data = response.json()
            formatter.success(data.get("message", "Review approved"))
        elif response.status_code == 404:
            formatter.error(f"Review not found: {review_id}")
            raise typer.Exit(1)
        else:
            error = response.json().get("error", {})
            formatter.error(
                error.get("message", "Failed to approve review"),
                code=error.get("code"),
            )
            raise typer.Exit(1)
            
    except httpx.RequestError as e:
        formatter.error(f"Connection error: {e}")
        raise typer.Exit(1)


@app.command("reject")
def reject_review(
    ctx: typer.Context,
    review_id: str = typer.Argument(
        ...,
        help="Review ID",
    ),
    reason: str = typer.Option(
        ...,
        "--reason", "-r",
        help="Reason for rejection",
    ),
):
    """
    Reject a review.
    
    Examples:
        dats review reject <review-id> --reason "Needs error handling"
    """
    formatter: Formatter = ctx.obj["formatter"]
    api_url: str = ctx.obj["api_url"]
    api_key: Optional[str] = ctx.obj["api_key"]
    
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["X-API-Key"] = api_key
    
    try:
        response = httpx.post(
            f"{api_url}/api/v1/reviews/{review_id}/reject",
            headers=headers,
            json={"reason": reason},
            timeout=30.0,
        )
        
        if response.status_code == 200:
            data = response.json()
            formatter.success(data.get("message", "Review rejected"))
        elif response.status_code == 404:
            formatter.error(f"Review not found: {review_id}")
            raise typer.Exit(1)
        else:
            error = response.json().get("error", {})
            formatter.error(
                error.get("message", "Failed to reject review"),
                code=error.get("code"),
            )
            raise typer.Exit(1)
            
    except httpx.RequestError as e:
        formatter.error(f"Connection error: {e}")
        raise typer.Exit(1)


@app.command("request-changes")
def request_changes(
    ctx: typer.Context,
    review_id: str = typer.Argument(
        ...,
        help="Review ID",
    ),
    guidance: str = typer.Option(
        ...,
        "--guidance", "-g",
        help="Guidance for changes",
    ),
    changes: Optional[str] = typer.Option(
        None,
        "--changes", "-c",
        help="Comma-separated list of required changes",
    ),
):
    """
    Request changes for a review.
    
    Examples:
        dats review request-changes <review-id> --guidance "Add input validation"
    """
    formatter: Formatter = ctx.obj["formatter"]
    api_url: str = ctx.obj["api_url"]
    api_key: Optional[str] = ctx.obj["api_key"]
    
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["X-API-Key"] = api_key
    
    payload = {"guidance": guidance}
    if changes:
        payload["required_changes"] = [c.strip() for c in changes.split(",")]
    
    try:
        response = httpx.post(
            f"{api_url}/api/v1/reviews/{review_id}/request-changes",
            headers=headers,
            json=payload,
            timeout=30.0,
        )
        
        if response.status_code == 200:
            data = response.json()
            formatter.success(data.get("message", "Changes requested"))
        elif response.status_code == 404:
            formatter.error(f"Review not found: {review_id}")
            raise typer.Exit(1)
        else:
            error = response.json().get("error", {})
            formatter.error(
                error.get("message", "Failed to request changes"),
                code=error.get("code"),
            )
            raise typer.Exit(1)
            
    except httpx.RequestError as e:
        formatter.error(f"Connection error: {e}")
        raise typer.Exit(1)