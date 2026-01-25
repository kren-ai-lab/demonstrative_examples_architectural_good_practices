from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field, ConfigDict

from mcs.provenance.record import ProvenanceRecord, ArtifactRef


NodeType = Literal["spec", "run", "artifact"]


class LineageNode(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(..., min_length=1)
    type: NodeType
    label: str = Field(..., min_length=1)
    payload: Dict[str, Any] = Field(default_factory=dict)


class LineageEdge(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: str = Field(..., min_length=1)
    target: str = Field(..., min_length=1)
    relation: Literal["uses", "produces"]


class LineageGraph(BaseModel):
    """
    Minimal lineage graph for paper-facing diagrams / debugging.

    Notes
    -----
    This is intentionally lightweight: a list of nodes + edges.
    It can be exported to JSON and later rendered (networkx, graphviz, etc.).
    """
    model_config = ConfigDict(extra="forbid")

    nodes: List[LineageNode] = Field(default_factory=list)
    edges: List[LineageEdge] = Field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump(mode="json", exclude_none=True)

    @classmethod
    def from_record(cls, record: ProvenanceRecord) -> "LineageGraph":
        g = cls()

        # Spec nodes (fingerprints as ids)
        spec_nodes = [
            ("dataset", record.dataset_fingerprint, record.dataset_name or "dataset"),
            ("split", record.split_fingerprint, record.split_name or "split"),
            ("embedding", record.embedding_fingerprint, record.embedding_name or "embedding"),
            ("train", record.train_fingerprint, record.train_name or "train"),
            ("execution", record.execution_fingerprint, record.execution_name or "execution"),
        ]
        for kind, fp, label in spec_nodes:
            g.nodes.append(
                LineageNode(
                    id=f"{kind}:{fp}",
                    type="spec",
                    label=f"{kind}:{label}",
                    payload={"fingerprint": fp},
                )
            )

        # Run node
        run_node_id = f"run:{record.run_id}"
        g.nodes.append(
            LineageNode(
                id=run_node_id,
                type="run",
                label=f"run:{record.run_id[:12]}",
                payload={
                    "created_at_utc": record.created_at_utc,
                    "mcs_version": record.mcs_version,
                },
            )
        )

        # Edges: run uses specs
        for kind, fp, _ in spec_nodes:
            g.edges.append(
                LineageEdge(
                    source=run_node_id,
                    target=f"{kind}:{fp}",
                    relation="uses",
                )
            )

        # Artifacts
        for a in record.artifacts:
            art_id = _artifact_node_id(record.run_id, a)
            g.nodes.append(
                LineageNode(
                    id=art_id,
                    type="artifact",
                    label=f"{a.kind}:{a.path}",
                    payload=a.model_dump(mode="json", exclude_none=True),
                )
            )
            g.edges.append(LineageEdge(source=run_node_id, target=art_id, relation="produces"))

        return g


def _artifact_node_id(run_id: str, a: ArtifactRef) -> str:
    # deterministic artifact id within the run
    base = f"{run_id}|{a.kind}|{a.path}|{a.format}|{a.checksum}"
    # use same hashing idea
    import hashlib
    h = hashlib.sha256(base.encode("utf-8")).hexdigest()
    return f"artifact:{h}"
