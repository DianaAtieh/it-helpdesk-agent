"""The LangGraph workflow. Phase 1: START -> check_status -> END.

The checkpointer saves the state after every step, keyed by the thread id the caller passes in
config={"configurable": {"thread_id": ...}}, so each conversation has its own history.
Later phases add nodes (triage, retrieve, answer, verify, escalate) and conditional edges here.
"""
from __future__ import annotations

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph

from app.agent.nodes import MCPClient, make_check_status
from app.agent.state import AgentState

# The node that ends every turn. Updates made outside a run (the approval gate saving its state,
# trimming old messages) are recorded as coming from it (aupdate_state(..., as_node=FINAL_NODE)),
# so nothing is left pending and the next turn starts from START. Keep it in step with the edges.
FINAL_NODE = "check_status"


def build_graph(checkpointer: BaseCheckpointSaver, mcp: MCPClient):
    graph = StateGraph(AgentState)
    graph.add_node("check_status", make_check_status(mcp))
    graph.add_edge(START, "check_status")
    graph.add_edge(FINAL_NODE, END)
    return graph.compile(checkpointer=checkpointer)
