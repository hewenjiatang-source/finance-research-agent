"""
DAG (directed acyclic graph) data structure and topological sort

The sub-task dependencies output by the planner are represented as a DAG, and the orchestrator schedules execution in topological order.
Uses Kahn's algorithm for topological sorting and supports grouping by layer to maximize parallelism.
"""
from __future__ import annotations

from collections import defaultdict, deque
from typing import Iterator


__all__ = ["DAG", "DAGCycleError"]


class DAGCycleError(Exception):
    """Raised when the DAG contains a cycle, indicating the planner output is invalid."""
    pass


class DAG:
    """Directed acyclic graph: nodes are task_ids, edges express dependencies (u -> v means v depends on u).

    Design notes:
      - adjacency-list storage, balancing memory efficiency and traversal speed
      - topological sort uses Kahn's algorithm, time complexity O(V+E)
      - get_parallel_groups() groups nodes into "execution layers"; nodes in one layer have no dependencies and can run in parallel
    """

    def __init__(self) -> None:
        self._nodes: set[str] = set()
        self._edges: dict[str, list[str]] = defaultdict(list)   # adjacency list: node -> successors
        self._in_degree: dict[str, int] = defaultdict(int)

    # ------------------------------------------------------------------
    # Add / remove / query
    # ------------------------------------------------------------------

    def add_node(self, node_id: str) -> None:
        """Add a node; silently ignored if it already exists."""
        self._nodes.add(node_id)

    def add_edge(self, from_node: str, to_node: str) -> None:
        """Add a directed edge from_node -> to_node (to_node depends on from_node).

        Missing nodes are added automatically and the in-degree is updated.
        """
        if from_node == to_node:
            raise DAGCycleError(f"Self-loops are not allowed: {from_node}")
        self._nodes.add(from_node)
        self._nodes.add(to_node)
        self._edges[from_node].append(to_node)
        self._in_degree[to_node] += 1
        # make sure from_node also has an entry in _in_degree (even if 0)
        self._in_degree.setdefault(from_node, 0)

    def has_node(self, node_id: str) -> bool:
        return node_id in self._nodes

    def get_dependencies(self, node_id: str) -> list[str]:
        """Return the list of nodes that directly depend on node_id (edges pointing out of node_id)."""
        deps: list[str] = []
        for src, dsts in self._edges.items():
            if node_id in dsts:
                deps.append(src)
        return deps

    def get_successors(self, node_id: str) -> list[str]:
        """Return the successor nodes node_id points to directly."""
        return list(self._edges.get(node_id, []))

    def __iter__(self) -> Iterator[str]:
        return iter(self._nodes)

    def __len__(self) -> int:
        return len(self._nodes)

    def __contains__(self, node_id: str) -> bool:
        return node_id in self._nodes

    # ------------------------------------------------------------------
    # Topological sort
    # ------------------------------------------------------------------

    def topological_sort(self) -> list[str]:
        """Kahn's algorithm topological sort, returning a full ordered list of nodes.

        Raises:
            DAGCycleError: raised when the graph contains a cycle.
        """
        in_deg = dict(self._in_degree)
        # add possibly missed nodes (isolated nodes have in-degree 0)
        for n in self._nodes:
            in_deg.setdefault(n, 0)

        queue: deque[str] = deque([n for n in self._nodes if in_deg.get(n, 0) == 0])
        result: list[str] = []

        while queue:
            node = queue.popleft()
            result.append(node)
            for succ in self._edges.get(node, []):
                in_deg[succ] -= 1
                if in_deg[succ] == 0:
                    queue.append(succ)

        if len(result) != len(self._nodes):
            # find the nodes on the cycle, for debugging
            remaining = self._nodes - set(result)
            raise DAGCycleError(
                f"The DAG contains a cycle, topological sort cannot complete. Remaining nodes: {sorted(remaining)}"
            )
        return result

    def get_parallel_groups(self) -> list[list[str]]:
        """Group by "execution layer", returning groups of nodes that can run in parallel.

        Nodes within a layer have no dependencies on each other and can be scheduled concurrently.
        The order of layers is the batches of the topological order.

        Returns:
            e.g. [["A", "B"], ["C"], ["D"]] means A/B in parallel, then C, then D.
        """
        in_deg = dict(self._in_degree)
        for n in self._nodes:
            in_deg.setdefault(n, 0)

        groups: list[list[str]] = []
        current: list[str] = [n for n in self._nodes if in_deg.get(n, 0) == 0]
        # stable sort in lexical order, to guarantee determinism
        current.sort()
        visited: set[str] = set()

        while current:
            groups.append(current)
            visited.update(current)
            next_layer: list[str] = []
            for node in current:
                for succ in self._edges.get(node, []):
                    in_deg[succ] -= 1
                    if in_deg[succ] == 0 and succ not in visited:
                        next_layer.append(succ)
            next_layer.sort()
            current = next_layer

        # safety check
        if sum(len(g) for g in groups) != len(self._nodes):
            raise DAGCycleError("The DAG contains a cycle, parallel groups cannot be computed")
        return groups

    def to_dict(self) -> dict:
        """Serialize to a dict, for logging and persistence."""
        return {
            "nodes": sorted(self._nodes),
            "edges": {k: sorted(v) for k, v in sorted(self._edges.items())},
        }

    def __repr__(self) -> str:
        return f"<DAG nodes={len(self._nodes)} edges={sum(len(v) for v in self._edges.values())}>"
