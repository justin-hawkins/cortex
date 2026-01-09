"""
Human review API routes.

Handles listing, viewing, and responding to human review requests.
"""

import logging
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status

from src.api.dependencies import get_review_queue, verify_api_key
from src.api.schemas import (
    AutomatedQAResult,
    ReviewActionResponse,
    ReviewApproveRequest,
    ReviewContext,
    ReviewListItem,
    ReviewListResponse,
    ReviewOption,
    ReviewRejectRequest,
    ReviewRequestChangesRequest,
    ReviewResponse,
    ReviewStatus,
    ReviewType,
)
from src.qa.human_review import (
    HumanReviewQueue,
    HumanReviewStatus,
    HumanReviewPriority,
)

logger = logging.getLogger(__name__)

router = APIRouter()


def _map_review_status(status: HumanReviewStatus) -> ReviewStatus:
    """Map internal review status to API status."""
    mapping = {
        HumanReviewStatus.PENDING: ReviewStatus.PENDING,
        HumanReviewStatus.IN_PROGRESS: ReviewStatus.PENDING,
        HumanReviewStatus.APPROVED: ReviewStatus.APPROVED,
        HumanReviewStatus.REJECTED: ReviewStatus.REJECTED,
        HumanReviewStatus.CHANGES_REQUESTED: ReviewStatus.CHANGES_REQUESTED,
        HumanReviewStatus.DELEGATED: ReviewStatus.PENDING,
        HumanReviewStatus.ESCALATED: ReviewStatus.PENDING,
        HumanReviewStatus.EXPIRED: ReviewStatus.REJECTED,
    }
    return mapping.get(status, ReviewStatus.PENDING)


def _map_review_type(output_type: str) -> ReviewType:
    """Map output type to review type."""
    if output_type in ("architecture", "design"):
        return ReviewType.ARCHITECTURE
    elif output_type == "ambiguity":
        return ReviewType.AMBIGUITY
    elif output_type == "approval":
        return ReviewType.APPROVAL
    return ReviewType.QA


@router.get(
    "/reviews",
    response_model=ReviewListResponse,
    summary="List pending reviews",
    description="List all pending human review requests",
)
async def list_reviews(
    project_id: Optional[str] = Query(default=None),
    priority: Optional[str] = Query(default=None),
    api_key: Optional[str] = Depends(verify_api_key),
):
    """List all pending human review requests."""
    review_queue = get_review_queue()
    
    # Map priority string to enum
    priority_enum = None
    if priority:
        try:
            priority_enum = HumanReviewPriority(priority.lower())
        except ValueError:
            pass
    
    # Get pending reviews
    pending = review_queue.get_pending_reviews(
        project_id=project_id,
        priority=priority_enum,
    )
    
    # Convert to response format
    reviews = []
    for request in pending:
        reviews.append(ReviewListItem(
            review_id=request.id,
            task_id=request.task_id,
            type=_map_review_type(request.output_type),
            status=_map_review_status(request.status),
            summary=request.task_description[:100] + "..." if len(request.task_description) > 100 else request.task_description,
            created_at=request.created_at or datetime.utcnow(),
            timeout_at=request.expires_at,
        ))
    
    return ReviewListResponse(
        reviews=reviews,
        total=len(reviews),
    )


@router.get(
    "/reviews/{review_id}",
    response_model=ReviewResponse,
    summary="Get review details",
    description="Get detailed information about a review request",
)
async def get_review(
    review_id: str,
    api_key: Optional[str] = Depends(verify_api_key),
):
    """Get detailed information about a review request."""
    review_queue = get_review_queue()
    
    request = review_queue.get_request(review_id)
    if not request:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "code": "REVIEW_NOT_FOUND",
                "message": f"Review with ID {review_id} not found",
            },
        )
    
    # Build context
    context = ReviewContext(
        task_description=request.task_description,
        output_summary=request.work_output[:500] if len(request.work_output) > 500 else request.work_output,
        specific_questions=request.questions_for_reviewer,
    )
    
    # Build automated results if available
    automated_results = None
    if request.automated_qa_results:
        automated_results = AutomatedQAResult(
            issues=request.automated_qa_results.get("issues", []),
            recommendations=request.automated_qa_results.get("recommendations", []),
        )
    
    # Build options from decision points
    options = []
    for i, point in enumerate(request.decision_points):
        options.append(ReviewOption(
            id=f"option_{i}",
            description=point,
            implications="",  # TODO: Add implications
        ))
    
    return ReviewResponse(
        review_id=request.id,
        task_id=request.task_id,
        type=_map_review_type(request.output_type),
        status=_map_review_status(request.status),
        context=context,
        automated_results=automated_results,
        options=options,
        recommendation=None,  # TODO: Generate recommendation
        created_at=request.created_at or datetime.utcnow(),
        timeout_at=request.expires_at,
    )


@router.post(
    "/reviews/{review_id}/approve",
    response_model=ReviewActionResponse,
    summary="Approve a review",
    description="Approve a pending review request",
)
async def approve_review(
    review_id: str,
    request: Optional[ReviewApproveRequest] = None,
    api_key: Optional[str] = Depends(verify_api_key),
):
    """Approve a pending review request."""
    review_queue = get_review_queue()
    
    review = review_queue.get_request(review_id)
    if not review:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "code": "REVIEW_NOT_FOUND",
                "message": f"Review with ID {review_id} not found",
            },
        )
    
    if review.status != HumanReviewStatus.PENDING:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "code": "REVIEW_NOT_PENDING",
                "message": f"Review is not pending (current status: {review.status.value})",
            },
        )
    
    # Submit the approval
    comments = request.comments if request else ""
    review_queue.submit_response(
        request_id=review_id,
        reviewer_id="api_user",  # TODO: Get from auth
        status=HumanReviewStatus.APPROVED,
        comments=comments or "",
    )
    
    logger.info(f"Review {review_id} approved")
    
    return ReviewActionResponse(
        review_id=review_id,
        status=ReviewStatus.APPROVED,
        message="Review approved successfully",
    )


@router.post(
    "/reviews/{review_id}/reject",
    response_model=ReviewActionResponse,
    summary="Reject a review",
    description="Reject a pending review request",
)
async def reject_review(
    review_id: str,
    request: ReviewRejectRequest,
    api_key: Optional[str] = Depends(verify_api_key),
):
    """Reject a pending review request."""
    review_queue = get_review_queue()
    
    review = review_queue.get_request(review_id)
    if not review:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "code": "REVIEW_NOT_FOUND",
                "message": f"Review with ID {review_id} not found",
            },
        )
    
    if review.status != HumanReviewStatus.PENDING:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "code": "REVIEW_NOT_PENDING",
                "message": f"Review is not pending (current status: {review.status.value})",
            },
        )
    
    # Submit the rejection
    review_queue.submit_response(
        request_id=review_id,
        reviewer_id="api_user",  # TODO: Get from auth
        status=HumanReviewStatus.REJECTED,
        comments=request.reason,
    )
    
    logger.info(f"Review {review_id} rejected: {request.reason}")
    
    return ReviewActionResponse(
        review_id=review_id,
        status=ReviewStatus.REJECTED,
        message="Review rejected",
    )


@router.post(
    "/reviews/{review_id}/request-changes",
    response_model=ReviewActionResponse,
    summary="Request changes",
    description="Request changes for a pending review",
)
async def request_changes(
    review_id: str,
    request: ReviewRequestChangesRequest,
    api_key: Optional[str] = Depends(verify_api_key),
):
    """Request changes for a pending review."""
    review_queue = get_review_queue()
    
    review = review_queue.get_request(review_id)
    if not review:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "code": "REVIEW_NOT_FOUND",
                "message": f"Review with ID {review_id} not found",
            },
        )
    
    if review.status != HumanReviewStatus.PENDING:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "code": "REVIEW_NOT_PENDING",
                "message": f"Review is not pending (current status: {review.status.value})",
            },
        )
    
    # Submit the changes request
    review_queue.submit_response(
        request_id=review_id,
        reviewer_id="api_user",  # TODO: Get from auth
        status=HumanReviewStatus.CHANGES_REQUESTED,
        comments=request.guidance,
        required_changes=request.required_changes,
    )
    
    logger.info(f"Changes requested for review {review_id}")
    
    return ReviewActionResponse(
        review_id=review_id,
        status=ReviewStatus.CHANGES_REQUESTED,
        message="Changes requested",
    )