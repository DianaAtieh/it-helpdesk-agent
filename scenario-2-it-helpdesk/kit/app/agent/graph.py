"""The LangGraph workflow (Phase 2):

    START -> triage --(a verified active incident)--> respond -> END
                    \\--(otherwise)--> retrieve -> answer -> respond

  triage    who is asking (directory), the status page (the ONLY status call of a turn), error codes, route
  retrieve  the knowledge base (ingestion's PolicyRetriever), scoped to the directory department
  answer    the one chat-model call: grounded in the evidence, no tools
  respond   the last check on everything that leaves the agent (citations, secrets, ticket IDs)

The checkpointer saves the state after every step, keyed by the thread id the caller passes in
config={"configurable": {"thread_id": ...}}, so each conversation has its own history.
No node can write a ticket: that stays in /api/approve, behind the human approval gate.
"""
from __future__ import annotations

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph

from app.agent.nodes import (MCPClient, make_answer, make_respond, make_retrieve, make_triage,
                             route_after_triage)
from app.agent.retrieval import KnowledgeBase
from app.agent.state import AgentState
from app.llm import ChatModel

# The node that ends every turn. Updates made outside a run (the approval gate saving its state,
# trimming old messages) are recorded as coming from it (aupdate_state(..., as_node=FINAL_NODE)),
# so nothing is left pending and the next turn starts from START. Keep it in step with the edges.
FINAL_NODE = "respond"


def build_graph(checkpointer: BaseCheckpointSaver, mcp: MCPClient, kb: KnowledgeBase | None = None,
                llm: ChatModel | None = None):
    """kb or llm may be None (not configured, or failed to load): the answer is then an honest error reply."""
    graph = StateGraph(AgentState)
    graph.add_node("triage", make_triage(mcp))
    graph.add_node("retrieve", make_retrieve(kb))
    graph.add_node("answer", make_answer(llm))
    graph.add_node(FINAL_NODE, make_respond())
    graph.add_edge(START, "triage")
    graph.add_conditional_edges("triage", route_after_triage, {"incident": FINAL_NODE, "retrieve": "retrieve"})
    graph.add_edge("retrieve", "answer")
    graph.add_edge("answer", FINAL_NODE)
    graph.add_edge(FINAL_NODE, END)
    return graph.compile(checkpointer=checkpointer)
