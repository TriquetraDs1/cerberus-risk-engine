"""Prometheus metrics for the /score endpoint — the "this team thinks about
production" signal a technical reviewer looks for and a bare hackathon script never
has. A separate registry (not the global default) keeps this importable in tests
without global state leaking between test runs.
"""

from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter, Histogram

registry = CollectorRegistry()

SCORE_REQUESTS = Counter(
    "cerberus_score_requests_total",
    "Total /score requests, by decision",
    ["decision"],
    registry=registry,
)

SCORE_LATENCY = Histogram(
    "cerberus_score_latency_seconds",
    "Time to score one transaction, end to end",
    registry=registry,
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
)

RING_CHECK_STATUS = Counter(
    "cerberus_ring_check_total",
    "Ring-detector availability at scoring time, by status",
    ["status"],
    registry=registry,
)
