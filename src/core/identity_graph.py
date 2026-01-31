"""
Identity Graph

A graph data structure representing resolved identities and their relationships.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any, Iterator
from datetime import datetime
import json
import hashlib

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from models.identity_entity import IdentityEntity, EntityType, PersonEntity, HouseholdEntity


@dataclass
class Edge:
    """An edge in the identity graph."""
    edge_id: str
    source_id: str
    target_id: str
    edge_type: str
    weight: float = 1.0
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: Optional[datetime] = None

    def __post_init__(self):
        if not self.edge_id:
            self.edge_id = hashlib.md5(
                f"{self.source_id}_{self.target_id}_{self.edge_type}".encode()
            ).hexdigest()[:12]
        if not self.created_at:
            self.created_at = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "edge_id": self.edge_id,
            "source_id": self.source_id,
            "target_id": self.target_id,
            "edge_type": self.edge_type,
            "weight": self.weight,
            "confidence": self.confidence,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class IdentityGraph:
    """
    A graph representing resolved identities and their relationships.

    Nodes are IdentityEntity objects (persons, devices, households).
    Edges represent relationships (uses_device, member_of, same_person).

    This is the core output of the identity resolution system.
    """

    # Edge types
    EDGE_USES_DEVICE = "uses_device"       # Person -> Device
    EDGE_MEMBER_OF = "member_of"           # Person -> Household
    EDGE_SAME_PERSON = "same_person"       # Person -> Person (cross-device link)
    EDGE_SAME_HOUSEHOLD = "same_household" # Household -> Household
    EDGE_HAS_SESSION = "has_session"       # Person -> Session
    EDGE_ACCOUNT_OF = "account_of"         # Account -> Household

    def __init__(self):
        """Initialize an empty identity graph."""
        self._nodes: Dict[str, IdentityEntity] = {}
        self._edges: Dict[str, Edge] = {}

        # Indexes for efficient lookup
        self._edges_by_source: Dict[str, List[str]] = {}
        self._edges_by_target: Dict[str, List[str]] = {}
        self._edges_by_type: Dict[str, List[str]] = {}
        self._nodes_by_type: Dict[EntityType, List[str]] = {}

    # =========================================================================
    # Node Operations
    # =========================================================================

    def add_node(self, entity: IdentityEntity) -> str:
        """
        Add a node to the graph.

        Parameters
        ----------
        entity : IdentityEntity
            The entity to add

        Returns
        -------
        str
            The entity ID
        """
        self._nodes[entity.entity_id] = entity

        # Update type index
        entity_type = entity.entity_type
        if entity_type not in self._nodes_by_type:
            self._nodes_by_type[entity_type] = []
        if entity.entity_id not in self._nodes_by_type[entity_type]:
            self._nodes_by_type[entity_type].append(entity.entity_id)

        return entity.entity_id

    def get_node(self, entity_id: str) -> Optional[IdentityEntity]:
        """Get a node by ID."""
        return self._nodes.get(entity_id)

    def remove_node(self, entity_id: str) -> bool:
        """
        Remove a node and all its edges.

        Returns
        -------
        bool
            True if node was removed
        """
        if entity_id not in self._nodes:
            return False

        # Remove edges
        edges_to_remove = list(self._edges_by_source.get(entity_id, []))
        edges_to_remove.extend(self._edges_by_target.get(entity_id, []))

        for edge_id in edges_to_remove:
            self.remove_edge(edge_id)

        # Remove from type index
        entity_type = self._nodes[entity_id].entity_type
        if entity_type in self._nodes_by_type:
            if entity_id in self._nodes_by_type[entity_type]:
                self._nodes_by_type[entity_type].remove(entity_id)

        # Remove node
        del self._nodes[entity_id]
        return True

    def get_nodes_by_type(self, entity_type: EntityType) -> List[IdentityEntity]:
        """Get all nodes of a specific type."""
        node_ids = self._nodes_by_type.get(entity_type, [])
        return [self._nodes[nid] for nid in node_ids if nid in self._nodes]

    def node_count(self) -> int:
        """Get total number of nodes."""
        return len(self._nodes)

    # =========================================================================
    # Edge Operations
    # =========================================================================

    def add_edge(
        self,
        source_id: str,
        target_id: str,
        edge_type: str,
        weight: float = 1.0,
        confidence: float = 1.0,
        **metadata
    ) -> str:
        """
        Add an edge between two nodes.

        Parameters
        ----------
        source_id : str
            Source node ID
        target_id : str
            Target node ID
        edge_type : str
            Type of relationship
        weight : float
            Edge weight (0-1)
        confidence : float
            Confidence in this edge (0-1)
        **metadata
            Additional edge metadata

        Returns
        -------
        str
            Edge ID
        """
        edge = Edge(
            edge_id="",
            source_id=source_id,
            target_id=target_id,
            edge_type=edge_type,
            weight=weight,
            confidence=confidence,
            metadata=metadata
        )

        self._edges[edge.edge_id] = edge

        # Update indexes
        if source_id not in self._edges_by_source:
            self._edges_by_source[source_id] = []
        self._edges_by_source[source_id].append(edge.edge_id)

        if target_id not in self._edges_by_target:
            self._edges_by_target[target_id] = []
        self._edges_by_target[target_id].append(edge.edge_id)

        if edge_type not in self._edges_by_type:
            self._edges_by_type[edge_type] = []
        self._edges_by_type[edge_type].append(edge.edge_id)

        return edge.edge_id

    def get_edge(self, edge_id: str) -> Optional[Edge]:
        """Get an edge by ID."""
        return self._edges.get(edge_id)

    def remove_edge(self, edge_id: str) -> bool:
        """Remove an edge."""
        if edge_id not in self._edges:
            return False

        edge = self._edges[edge_id]

        # Update indexes
        if edge.source_id in self._edges_by_source:
            if edge_id in self._edges_by_source[edge.source_id]:
                self._edges_by_source[edge.source_id].remove(edge_id)

        if edge.target_id in self._edges_by_target:
            if edge_id in self._edges_by_target[edge.target_id]:
                self._edges_by_target[edge.target_id].remove(edge_id)

        if edge.edge_type in self._edges_by_type:
            if edge_id in self._edges_by_type[edge.edge_type]:
                self._edges_by_type[edge.edge_type].remove(edge_id)

        del self._edges[edge_id]
        return True

    def get_edges_from(self, node_id: str) -> List[Edge]:
        """Get all edges originating from a node."""
        edge_ids = self._edges_by_source.get(node_id, [])
        return [self._edges[eid] for eid in edge_ids if eid in self._edges]

    def get_edges_to(self, node_id: str) -> List[Edge]:
        """Get all edges pointing to a node."""
        edge_ids = self._edges_by_target.get(node_id, [])
        return [self._edges[eid] for eid in edge_ids if eid in self._edges]

    def get_edges_by_type(self, edge_type: str) -> List[Edge]:
        """Get all edges of a specific type."""
        edge_ids = self._edges_by_type.get(edge_type, [])
        return [self._edges[eid] for eid in edge_ids if eid in self._edges]

    def edge_count(self) -> int:
        """Get total number of edges."""
        return len(self._edges)

    # =========================================================================
    # Graph Queries
    # =========================================================================

    def get_neighbors(self, node_id: str) -> List[str]:
        """Get all neighbor node IDs (both directions)."""
        neighbors = set()

        for edge in self.get_edges_from(node_id):
            neighbors.add(edge.target_id)

        for edge in self.get_edges_to(node_id):
            neighbors.add(edge.source_id)

        return list(neighbors)

    def get_connected_component(self, node_id: str) -> Set[str]:
        """
        Get all nodes in the same connected component.

        Uses BFS to find all reachable nodes.
        """
        if node_id not in self._nodes:
            return set()

        visited = set()
        queue = [node_id]

        while queue:
            current = queue.pop(0)
            if current in visited:
                continue

            visited.add(current)

            for neighbor_id in self.get_neighbors(current):
                if neighbor_id not in visited:
                    queue.append(neighbor_id)

        return visited

    def get_all_connected_components(self) -> List[Set[str]]:
        """Get all connected components in the graph."""
        visited = set()
        components = []

        for node_id in self._nodes:
            if node_id not in visited:
                component = self.get_connected_component(node_id)
                visited.update(component)
                components.append(component)

        return components

    def get_persons_for_device(self, device_id: str) -> List[Tuple[str, float]]:
        """
        Get persons who use a device with their probabilities.

        Returns
        -------
        List[Tuple[str, float]]
            List of (person_id, probability) tuples
        """
        results = []

        for edge in self.get_edges_to(device_id):
            if edge.edge_type == self.EDGE_USES_DEVICE:
                results.append((edge.source_id, edge.weight))

        return results

    def get_devices_for_person(self, person_id: str) -> List[Tuple[str, float]]:
        """
        Get devices used by a person with their affinities.

        Returns
        -------
        List[Tuple[str, float]]
            List of (device_id, affinity) tuples
        """
        results = []

        for edge in self.get_edges_from(person_id):
            if edge.edge_type == self.EDGE_USES_DEVICE:
                results.append((edge.target_id, edge.weight))

        return results

    def get_household_members(self, household_id: str) -> List[str]:
        """Get all person IDs who are members of a household."""
        members = []

        for edge in self.get_edges_to(household_id):
            if edge.edge_type == self.EDGE_MEMBER_OF:
                members.append(edge.source_id)

        return members

    # =========================================================================
    # Merge Operations
    # =========================================================================

    def merge_nodes(self, keep_id: str, remove_id: str) -> bool:
        """
        Merge two nodes into one.

        All edges of the removed node are transferred to the kept node.
        Used when we determine two entities are actually the same.

        Parameters
        ----------
        keep_id : str
            Node ID to keep
        remove_id : str
            Node ID to remove (merge into keep)

        Returns
        -------
        bool
            True if merge was successful
        """
        if keep_id not in self._nodes or remove_id not in self._nodes:
            return False

        keep_node = self._nodes[keep_id]
        remove_node = self._nodes[remove_id]

        # Merge node data
        keep_node.merge_with(remove_node)

        # Transfer edges
        for edge in self.get_edges_from(remove_id):
            if edge.target_id != keep_id:  # Avoid self-loops
                self.add_edge(
                    keep_id, edge.target_id, edge.edge_type,
                    edge.weight, edge.confidence, **edge.metadata
                )

        for edge in self.get_edges_to(remove_id):
            if edge.source_id != keep_id:  # Avoid self-loops
                self.add_edge(
                    edge.source_id, keep_id, edge.edge_type,
                    edge.weight, edge.confidence, **edge.metadata
                )

        # Remove the merged node
        self.remove_node(remove_id)

        return True

    # =========================================================================
    # Serialization
    # =========================================================================

    def to_dict(self) -> Dict[str, Any]:
        """Serialize graph to dictionary."""
        return {
            "nodes": [node.to_dict() for node in self._nodes.values()],
            "edges": [edge.to_dict() for edge in self._edges.values()],
            "metadata": {
                "node_count": self.node_count(),
                "edge_count": self.edge_count(),
                "created_at": datetime.now().isoformat(),
            }
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "IdentityGraph":
        """Deserialize graph from dictionary."""
        graph = cls()

        for node_data in data.get("nodes", []):
            entity = IdentityEntity.from_dict(node_data)
            graph.add_node(entity)

        for edge_data in data.get("edges", []):
            graph.add_edge(
                source_id=edge_data["source_id"],
                target_id=edge_data["target_id"],
                edge_type=edge_data["edge_type"],
                weight=edge_data.get("weight", 1.0),
                confidence=edge_data.get("confidence", 1.0),
                **edge_data.get("metadata", {})
            )

        return graph

    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)

    @classmethod
    def from_json(cls, json_str: str) -> "IdentityGraph":
        """Deserialize from JSON string."""
        return cls.from_dict(json.loads(json_str))

    # =========================================================================
    # Statistics and Summary
    # =========================================================================

    def get_statistics(self) -> Dict[str, Any]:
        """Get graph statistics."""
        stats = {
            "total_nodes": self.node_count(),
            "total_edges": self.edge_count(),
            "nodes_by_type": {},
            "edges_by_type": {},
            "connected_components": len(self.get_all_connected_components()),
        }

        for entity_type in EntityType:
            nodes = self.get_nodes_by_type(entity_type)
            stats["nodes_by_type"][entity_type.value] = len(nodes)

        for edge_type, edge_ids in self._edges_by_type.items():
            stats["edges_by_type"][edge_type] = len(edge_ids)

        return stats

    def __repr__(self) -> str:
        stats = self.get_statistics()
        return (
            f"IdentityGraph(nodes={stats['total_nodes']}, "
            f"edges={stats['total_edges']}, "
            f"components={stats['connected_components']})"
        )

    def __iter__(self) -> Iterator[IdentityEntity]:
        """Iterate over all nodes."""
        return iter(self._nodes.values())

    def __len__(self) -> int:
        """Return number of nodes."""
        return self.node_count()
