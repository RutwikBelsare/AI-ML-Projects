
from app.agent.graph import research_graph


def test_graph_contains_required_nodes():
    """The compiled graph should contain all Phase 2 nodes."""

    graph = research_graph.get_graph()
    node_names = set(graph.nodes.keys())

    assert "planner" in node_names
    assert "agent" in node_names
    assert "tools" in node_names
    assert "finalize" in node_names


def test_graph_has_expected_connections():
    """The graph should include the RAG retrieval workflow."""

    graph = research_graph.get_graph()

    edges = {
        (edge.source, edge.target)
        for edge in graph.edges
    }

    # Entry and RAG retrieval path
    assert ("__start__", "planner") in edges
    assert ("planner", "retrieve") in edges
    assert ("retrieve", "agent") in edges
    assert ("retrieve", "finalize") in edges

    # Existing agent and tool loop
    assert ("agent", "tools") in edges
    assert ("tools", "agent") in edges
    assert ("agent", "finalize") in edges

    # Finalization
    assert ("finalize", "__end__") in edges