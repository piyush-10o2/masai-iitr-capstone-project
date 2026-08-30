"""
app/models.py — shared Pydantic schemas.

These models double as: (1) the CrewAI structured-output target
(`response_format`) for Task 9, (2) the Autogen verdict model for Task 14, and
(3) the FastAPI request/response models for Task 11 — one schema definition,
several jobs, so we don't maintain parallel schemas across three frameworks.
"""

from enum import Enum
from typing import List, Literal, Optional

from pydantic import BaseModel, Field


# ── Dataset-mirroring enums (kept in sync with dataset.py's CATEGORIES/STATUSES) ──

class TicketCategory(str, Enum):
    BILLING = "Billing"
    TECHNICAL_ISSUE = "Technical Issue"
    ACCOUNT_ACCESS = "Account Access"
    PRODUCT_DEFECT = "Product Defect"
    GENERAL_INQUIRY = "General Inquiry"


class TicketStatus(str, Enum):
    OPEN = "Open"
    IN_PROGRESS = "In Progress"
    ESCALATED = "Escalated"
    RESOLVED = "Resolved"
    CLOSED = "Closed"


class SupportTicket(BaseModel):
    """Validated/typed mirror of dataset.py's SupportTicket dataclass, for use
    anywhere a ticket needs to cross a function or API boundary."""
    record_id: str
    category: TicketCategory
    status: TicketStatus
    resolution_time_hours: int = Field(ge=0)
    days_since_created: int = Field(ge=0, le=30)
    escalated: bool


# ── Task 6 — lookup tool output ─────────────────────────────────────────────

class TicketLookupResult(BaseModel):
    """Return type of check_support_ticket_status()."""
    record_id: str
    status: TicketStatus
    resolution_time_hours: int
    escalation_score: float = Field(ge=0.0, le=1.0)
    escalation_recommended: bool


# ── Task 9 — CrewAI structured output ───────────────────────────────────────

class AgentResponse(BaseModel):
    """Every crew response (the Response Composer's final draft, Task 7) must
    validate against this schema before being handed to the Autogen review
    stage (Task 14)."""
    answer: str
    source: Literal["rag", "lookup", "both", "refused"]
    ticket_id: Optional[str] = None
    escalation_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    grounded: bool
    retrieved_chunk_ids: List[str] = Field(default_factory=list)


# ── Task 14 — Autogen review-stage verdict ──────────────────────────────────

class ReviewVerdict(BaseModel):
    """Structured output required from the Final-Editor agent via
    output_content_type=ReviewVerdict on the Autogen agent (and
    custom_message_types=[StructuredMessage[ReviewVerdict]] on the team)."""
    approved: bool
    final_answer: str
    reason: str


# ── Task 11 — FastAPI request/response models ───────────────────────────────

class AskRequest(BaseModel):
    query: str
    ticket_id: Optional[str] = None


class AskResponse(BaseModel):
    agent_response: AgentResponse
    review: ReviewVerdict


class AddDocumentRequest(BaseModel):
    topic: str
    content: str


class AddDocumentResponse(BaseModel):
    document_id: str
    chunks_indexed_fixed: int
    chunks_indexed_sentence: int