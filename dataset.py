"""
dataset.py — Task 1: Ola Support-Ticket dataset generator.

Generates a deterministic, seeded list of `SUPPORT_TICKETS` used throughout the
project (Task 6 lookup tool, Task 13 evaluation set, etc.). Running this file
directly regenerates the dataset and prints the structural report required by
the brief (category counts, status counts, escalated percentage).

Reproduce with:  python dataset.py --seed 42
"""

import argparse
import random
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional

CATEGORIES = ["Billing", "Technical Issue", "Account Access", "Product Defect", "General Inquiry"]
STATUSES = ["Open", "In Progress", "Escalated", "Resolved", "Closed"]

MIN_RECORDS = 40
DEFAULT_RECORD_COUNT = 50
MIN_PER_CATEGORY = 3
MIN_PER_STATUS = 1
ESCALATED_PCT_MIN = 0.10
ESCALATED_PCT_MAX = 0.30

# Resolution-time range (hours). Chosen to span Ola's SLA tiers (see
# data/knowledge_base/sla_by_severity.md): fast resolutions for low-severity
# General Inquiry / Billing tickets through the slowest permitted window for
# escalated Technical Issue / Product Defect tickets — so the generated
# dataset contains both comfortably-on-SLA and SLA-breaching tickets, which
# Task 6's escalation scoring needs to be meaningful.
RESOLUTION_TIME_MIN_HOURS = 1
RESOLUTION_TIME_MAX_HOURS = 96

DAYS_SINCE_CREATED_MIN = 0
DAYS_SINCE_CREATED_MAX = 30

# Targets the middle of the required 10-30% escalated band.
ESCALATED_PROBABILITY = 0.20


@dataclass
class SupportTicket:
    record_id: str
    category: str
    status: str
    resolution_time_hours: int
    days_since_created: int
    escalated: bool

    def to_dict(self) -> Dict:
        return asdict(self)


def _generate_once(seed: int, n: int) -> List[SupportTicket]:
    rng = random.Random(seed)

    # Guarantee every category gets >= MIN_PER_CATEGORY and every status gets
    # >= MIN_PER_STATUS by seeding the pool directly, then filling the
    # remainder randomly and shuffling. This keeps the *coverage* thresholds
    # unconditionally true, independent of the escalated-percentage check below.
    forced_categories = CATEGORIES * MIN_PER_CATEGORY
    remaining = n - len(forced_categories)
    if remaining < 0:
        raise ValueError("n too small to satisfy minimum category coverage")
    category_pool = forced_categories + [rng.choice(CATEGORIES) for _ in range(remaining)]
    rng.shuffle(category_pool)

    forced_statuses = STATUSES * MIN_PER_STATUS
    status_pool = forced_statuses + [rng.choice(STATUSES) for _ in range(n - len(forced_statuses))]
    rng.shuffle(status_pool)

    tickets: List[SupportTicket] = []
    for i in range(n):
        record_id = f"TCK-{i + 1:04d}"
        category = category_pool[i]
        status = status_pool[i]
        resolution_time_hours = rng.randint(RESOLUTION_TIME_MIN_HOURS, RESOLUTION_TIME_MAX_HOURS)
        days_since_created = rng.randint(DAYS_SINCE_CREATED_MIN, DAYS_SINCE_CREATED_MAX)
        escalated = rng.random() < ESCALATED_PROBABILITY

        tickets.append(SupportTicket(
            record_id=record_id,
            category=category,
            status=status,
            resolution_time_hours=resolution_time_hours,
            days_since_created=days_since_created,
            escalated=escalated,
        ))
    return tickets


def _satisfies_constraints(tickets: List[SupportTicket]) -> bool:
    if len(tickets) < MIN_RECORDS:
        return False
    for cat in CATEGORIES:
        if sum(1 for t in tickets if t.category == cat) < MIN_PER_CATEGORY:
            return False
    for st in STATUSES:
        if sum(1 for t in tickets if t.status == st) < MIN_PER_STATUS:
            return False
    escalated_pct = sum(1 for t in tickets if t.escalated) / len(tickets)
    return ESCALATED_PCT_MIN <= escalated_pct <= ESCALATED_PCT_MAX


def generate_dataset(seed: int = 42, n: int = DEFAULT_RECORD_COUNT,
                      max_seed_attempts: int = 500) -> "tuple[List[SupportTicket], int]":
    """
    Deterministically generate the ticket dataset. If the requested seed
    doesn't happen to satisfy every structural threshold (category/status
    coverage, escalated-percentage band), this automatically tries subsequent
    seeds (seed, seed+1, seed+2, ...) rather than hand-editing records — the
    automated version of the brief's "change your seed and regenerate"
    instruction. Returns (tickets, seed_actually_used).
    """
    for attempt in range(max_seed_attempts):
        candidate_seed = seed + attempt
        tickets = _generate_once(candidate_seed, n)
        if _satisfies_constraints(tickets):
            return tickets, candidate_seed
    raise RuntimeError(
        f"Could not find a satisfying seed within {max_seed_attempts} attempts "
        f"starting from {seed}. Consider adjusting ESCALATED_PROBABILITY."
    )


def print_report(tickets: List[SupportTicket], seed_requested: int, seed_used: int) -> None:
    n = len(tickets)
    if seed_used != seed_requested:
        print(f"[dataset.py] Seed {seed_requested} did not satisfy all constraints; "
              f"used seed {seed_used} instead.")
    print(f"Seed used: {seed_used}")
    print(f"Total records: {n}")
    print("---")
    for cat in CATEGORIES:
        count = sum(1 for t in tickets if t.category == cat)
        print(f"{cat}: {count}")
    print("---")
    for st in STATUSES:
        count = sum(1 for t in tickets if t.status == st)
        print(f"{st}: {count}")
    print("---")
    escalated_count = sum(1 for t in tickets if t.escalated)
    pct = 100 * escalated_count / n
    print(f"escalated=True: {pct:.1f}% ({escalated_count}/{n})")


def get_ticket_by_id(tickets: List[SupportTicket], record_id: str) -> Optional[SupportTicket]:
    for t in tickets:
        if t.record_id == record_id:
            return t
    return None


# Module-level constant used everywhere else in the project (e.g.
# app.tools.ticket_lookup_tool imports SUPPORT_TICKETS directly). Generated
# once, at import time, with the project's documented default seed.
SUPPORT_TICKETS, SEED_USED = generate_dataset(seed=42, n=DEFAULT_RECORD_COUNT)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate the Ola SUPPORT_TICKETS dataset.")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed (default: 42)")
    parser.add_argument("--n", type=int, default=DEFAULT_RECORD_COUNT, help="Number of records (default: 50, min 40)")
    args = parser.parse_args()

    tickets, actual_seed = generate_dataset(seed=args.seed, n=args.n)
    print_report(tickets, seed_requested=args.seed, seed_used=actual_seed)