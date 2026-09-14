"""Local web UI for browsing and managing Poppy memories.

Launched via `poppy ui`. FastAPI server bound to localhost only; static frontend
served from `static/`. Writes go through the existing engine + a sidecar
tombstone table so the immutable RetrievalEngine ABC stays untouched.
"""
