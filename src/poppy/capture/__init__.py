"""Auto-capture deep modules (ADR-0001/0002/0003).

Each module here is a small, independently testable unit that the capture
orchestration composes:

* ``reconciler``  — dedup-on-capture: ADD / SUPERSEDE / SKIP before ingest (ADR-0003).
* ``window``      — incremental transcript window over ``(watermark, now]`` (ADR-0001).
* ``watermark``   — per-session capture watermark + turn cadence (ADR-0001).

The SessionEnd / PostCompact backstops and the mid-session loop all route their
extracted candidates through ``reconciler.reconcile_and_ingest`` so the store
stays clean no matter what triggered the capture.
"""
