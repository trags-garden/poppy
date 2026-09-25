"""Auto-capture deep modules.

This package is governed by three rules: watermark-after-success capture,
consent and default-on precedence, and automatic candidate reconciliation.

Each module here is a small, independently testable unit that the capture
orchestration composes:

* ``reconciler``  — dedup-on-capture: ADD / SUPERSEDE / SKIP before ingest
  (the automatic reconciliation rule).
* ``window``      — incremental transcript window over ``(watermark, now]``
  (the watermark-after-success rule).
* ``watermark``   — per-session capture watermark + turn cadence (the
  watermark-after-success rule).

The SessionEnd / PostCompact backstops and the mid-session loop all route their
extracted candidates through ``reconciler.reconcile_and_ingest`` so the store
stays clean no matter what triggered the capture.
"""
