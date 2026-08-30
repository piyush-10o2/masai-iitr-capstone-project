"""
app/tools/ticket_lookup_tool.py — Task 6: check_support_ticket_status.

Looks up a ticket by record_id in the generated SUPPORT_TICKETS dataset and
returns its status, resolution_time_hours, and a designed escalation_score.

Formula (see app/config.py for the constants and full rationale):
    escalation_score = ESCALATION_WEIGHT_FLAGGED   * float(escalated)
                      + ESCALATION_WEIGHT_RECENCY  * min(days_since_created / P_k, 1.0)

P_k (the ESCALATION_RECENCY_PERCENTILE-th percentile of days_since_created) is
recomputed from the *live* dataset at call time — not hardcoded — so the
formula stays correct even if dataset.py is regenerated with a different seed.
"""

from typing import Optional

from dataset import SUPPORT_TICKETS, get_ticket_by_id
from app.config import (
    ESCALATION_WEIGHT_FLAGGED,
    ESCALATION_WEIGHT_RECENCY,
    ESCALATION_RECENCY_PERCENTILE,
    ESCALATION_SCORE_THRESHOLD,
)
from app.models import TicketLookupResult, TicketStatus


def _percentile(values, pct: int) -> float:
    """Nearest-rank percentile, dependency-free (no numpy needed for this)."""
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    idx = min(int(len(sorted_vals) * pct / 100), len(sorted_vals) - 1)
    return float(sorted_vals[idx])


def _recency_percentile_value() -> float:
    """P_k of days_since_created across the current SUPPORT_TICKETS dataset."""
    days = [t.days_since_created for t in SUPPORT_TICKETS]
    return _percentile(days, ESCALATION_RECENCY_PERCENTILE)


def check_support_ticket_status(record_id: str) -> Optional[TicketLookupResult]:
    """
    Task 6. Returns the ticket's status, resolution_time_hours, and a designed
    escalation_score in [0, 1]. Returns None if record_id is not found — the
    calling agent should surface "ticket not found" rather than treat this as
    an error to hide.
    """
    ticket = get_ticket_by_id(SUPPORT_TICKETS, record_id)
    if ticket is None:
        return None

    p_k = _recency_percentile_value()
    recency_signal = min(ticket.days_since_created / p_k, 1.0) if p_k > 0 else 0.0

    escalation_score = (
        ESCALATION_WEIGHT_FLAGGED * float(ticket.escalated)
        + ESCALATION_WEIGHT_RECENCY * recency_signal
    )
    escalation_score = round(min(max(escalation_score, 0.0), 1.0), 4)

    return TicketLookupResult(
        record_id=ticket.record_id,
        status=TicketStatus(ticket.status),
        resolution_time_hours=ticket.resolution_time_hours,
        escalation_score=escalation_score,
        escalation_recommended=escalation_score > ESCALATION_SCORE_THRESHOLD,
    )


if __name__ == "__main__":
    # Manual smoke test — run: python -m app.tools.ticket_lookup_tool
    for sample_id in ["TCK-0001", "TCK-0002", "TCK-0010", "TCK-9999"]:
        print(sample_id, "->", check_support_ticket_status(sample_id))