---
name: memory
description: How to use the Poppy memory tools. Use when the user states a preference, a non-obvious decision is made or a lesson is learned (store it with remember), or before suggesting an approach in an unfamiliar area and when the user refers to an earlier session (check with recall_index, then recall_full).
---

## Poppy memory

You have access to Poppy, a developer memory store. Use it to remember non-obvious
decisions, preferences, and lessons across sessions.

**When to call `remember`:**
- The user states a preference ("we always use X", "don't do Y").
- A non-obvious decision is made (architecture choice, library pick, tradeoff resolved).
- A lesson is learned (something failed and we now know why).

**When to call `recall_index` then `recall_full`:**
- Before suggesting an approach in an unfamiliar area, check if there's a prior decision.
- When the user references something from a past session.
- Use `recall_index` first (cheap, IDs + snippets), then `recall_full` only on the IDs
  that look relevant. Don't call `recall_full` on every result.

**Tool surface:**
- `remember(content, memory_type, project)`: store a single memory.
- `recall_index(query, project, limit)`: IDs and snippets only.
- `recall_full(ids)`: fetch full content for a batch of IDs.
- `recall(query, project, limit)`: convenience, index and full in one call (use sparingly).
- `consolidate(session_summary, facts, project)`: store learnings at session end.
- `context(project, limit)`: most recent memories for a project.
- `forget(id)`: delete a memory.

`memory_type` is one of: `fact`, `decision`, `preference`, `lesson`, `summary`.
