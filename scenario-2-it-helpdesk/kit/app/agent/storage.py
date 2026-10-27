"""Keeping the checkpoint database small and short-lived.

Why: the SQLite saver stores the WHOLE state (every message) in every checkpoint, and each turn writes
about three checkpoints, so without this a conversation's storage grows with the square of its length.
  * trim_history: keep only the last MAX_HISTORY_MESSAGES messages of a conversation;
  * prune_thread: after a turn, keep only the newest checkpoint of the conversation;
  * delete_expired_threads: forget conversations that have been idle for CHECKPOINT_RETENTION_DAYS.

prune_thread works on the saver's tables directly: AsyncSqliteSaver.aprune() is not implemented in
langgraph-checkpoint-sqlite 3.1.2 (it raises NotImplementedError). The SQL uses the saver's own lock and
connection, and only deletes rows of one thread in the root namespace (this graph has no subgraphs).
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone

from langchain_core.messages import RemoveMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from app.agent.graph import FINAL_NODE


def max_history_messages() -> int:
    return int(os.getenv("MAX_HISTORY_MESSAGES") or 40)


def retention_days() -> int:
    return int(os.getenv("CHECKPOINT_RETENTION_DAYS") or 30)


async def trim_history(graph, config: dict, limit: int | None = None) -> int:
    """Remove the oldest messages above the limit. Returns how many were removed."""
    limit = max_history_messages() if limit is None else limit
    messages = (await graph.aget_state(config)).values.get("messages", [])
    extra = len(messages) - limit
    if extra <= 0:
        return 0
    await graph.aupdate_state(config, {"messages": [RemoveMessage(id=m.id) for m in messages[:extra]]},
                              as_node=FINAL_NODE)
    return extra


async def prune_thread(checkpointer: AsyncSqliteSaver, thread_id: str) -> int:
    """Delete every checkpoint of the thread except the newest one (and their writes). Returns rows deleted."""
    newest = ("SELECT checkpoint_id FROM checkpoints WHERE thread_id = ? AND checkpoint_ns = '' "
              "ORDER BY checkpoint_id DESC LIMIT 1")
    async with checkpointer.lock, checkpointer.conn.cursor() as cur:
        await cur.execute(f"DELETE FROM writes WHERE thread_id = ? AND checkpoint_ns = '' "
                          f"AND checkpoint_id NOT IN ({newest})", (thread_id, thread_id))
        await cur.execute(f"DELETE FROM checkpoints WHERE thread_id = ? AND checkpoint_ns = '' "
                          f"AND checkpoint_id NOT IN ({newest})", (thread_id, thread_id))
        deleted = cur.rowcount
        await checkpointer.conn.commit()
    return deleted


async def delete_expired_threads(checkpointer: AsyncSqliteSaver, days: int | None = None) -> int:
    """Delete conversations whose newest checkpoint is older than `days` days. Returns how many."""
    days = retention_days() if days is None else days
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    newest: dict[str, datetime] = {}
    async for item in checkpointer.alist(None):
        thread_id = item.config["configurable"]["thread_id"]
        saved_at = datetime.fromisoformat(item.checkpoint["ts"])
        if thread_id not in newest or saved_at > newest[thread_id]:
            newest[thread_id] = saved_at
    expired = [thread_id for thread_id, saved_at in newest.items() if saved_at < cutoff]
    for thread_id in expired:
        await checkpointer.adelete_thread(thread_id)
    return len(expired)
