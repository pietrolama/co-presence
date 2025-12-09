"""
RAG Store: Semantic memory for each agent
"""
import json
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings

from ..environment.artifact import Artifact


class RAGStore:
    """
    Individual RAG store for an agent.
    - Vector store for semantic retrieval
    - Indexes artifacts, concepts, and world summaries
    - Supports multi-dimensional retrieval (not just similarity)
    """
    
    def __init__(self, agent_name: str, storage_path: Path):
        self.agent_name = agent_name
        self.storage_path = storage_path
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize ChromaDB with persistent storage
        self.client = chromadb.Client(Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory=str(storage_path),
            anonymized_telemetry=False,
        ))
        
        # Collections for different content types
        self.artifacts_collection = self.client.get_or_create_collection(
            name=f"{agent_name.replace(' ', '_').lower()}_artifacts",
            metadata={"description": "Indexed artifacts from Environment"}
        )
        
        self.concepts_collection = self.client.get_or_create_collection(
            name=f"{agent_name.replace(' ', '_').lower()}_concepts",
            metadata={"description": "Internal concepts created by agent"}
        )
        
        self.world_summaries_collection = self.client.get_or_create_collection(
            name=f"{agent_name.replace(' ', '_').lower()}_world",
            metadata={"description": "Summaries of World content"}
        )
    
    def index_artifact(self, artifact: Artifact) -> None:
        """Index an artifact for later retrieval"""
        # Create text representation for embedding
        text_parts = [
            artifact.artifact.description,
            " ".join(s.content for s in artifact.artifact.steps),
            artifact.artifact.meta_cognition.self_observation,
            artifact.artifact.meta_cognition.influence_of_other_agent,
            artifact.artifact.meta_cognition.uncertainties,
        ]
        text = " ".join(text_parts)
        
        # Metadata for filtering
        metadata = {
            "agent_name": artifact.agent_name,
            "cycle_id": artifact.cycle_id,
            "artifact_type": artifact.artifact_type.value,
            "silence_flag": artifact.silence_flag,
            "has_profile_update": bool(artifact.profile_update and artifact.profile_update.proposed_changes),
            "has_uncertainty": bool(artifact.artifact.meta_cognition.uncertainties.strip()),
            "step_count": len(artifact.artifact.steps),
        }
        
        self.artifacts_collection.add(
            documents=[text],
            metadatas=[metadata],
            ids=[artifact.id],
        )
    
    def add_concept(self, concept_id: str, content: str, metadata: dict[str, Any] | None = None) -> None:
        """Add an internal concept created by the agent"""
        self.concepts_collection.add(
            documents=[content],
            metadatas=[metadata or {}],
            ids=[concept_id],
        )
    
    def add_world_summary(self, summary_id: str, content: str, source_ids: list[str]) -> None:
        """Add a summary of World content"""
        self.world_summaries_collection.add(
            documents=[content],
            metadatas=[{"source_ids": json.dumps(source_ids)}],
            ids=[summary_id],
        )
    
    def query_similar(
        self,
        query_text: str,
        collection: str = "artifacts",
        n_results: int = 5,
        where: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Query for similar content.
        Supports filtering by metadata.
        """
        coll = {
            "artifacts": self.artifacts_collection,
            "concepts": self.concepts_collection,
            "world": self.world_summaries_collection,
        }.get(collection, self.artifacts_collection)
        
        results = coll.query(
            query_texts=[query_text],
            n_results=n_results,
            where=where,
        )
        
        # Format results
        formatted = []
        if results and results["documents"]:
            for i, doc in enumerate(results["documents"][0]):
                formatted.append({
                    "id": results["ids"][0][i] if results["ids"] else None,
                    "document": doc,
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                    "distance": results["distances"][0][i] if results.get("distances") else None,
                })
        
        return formatted
    
    def query_by_metadata(
        self,
        collection: str = "artifacts",
        where: dict[str, Any] | None = None,
        n_results: int = 10,
    ) -> list[dict[str, Any]]:
        """
        Query by metadata filters without semantic similarity.
        """
        coll = {
            "artifacts": self.artifacts_collection,
            "concepts": self.concepts_collection,
            "world": self.world_summaries_collection,
        }.get(collection, self.artifacts_collection)
        
        results = coll.get(
            where=where,
            limit=n_results,
        )
        
        formatted = []
        if results and results["documents"]:
            for i, doc in enumerate(results["documents"]):
                formatted.append({
                    "id": results["ids"][i] if results["ids"] else None,
                    "document": doc,
                    "metadata": results["metadatas"][i] if results["metadatas"] else {},
                })
        
        return formatted
    
    def get_recent_artifacts(
        self,
        agent_name: str | None = None,
        n_results: int = 5,
    ) -> list[dict[str, Any]]:
        """Get recent artifacts, optionally filtered by agent"""
        where = {"agent_name": agent_name} if agent_name else None
        
        results = self.artifacts_collection.get(
            where=where,
            limit=n_results,
        )
        
        # Sort by cycle_id (descending) - need to do this manually
        items = []
        if results and results["documents"]:
            for i, doc in enumerate(results["documents"]):
                items.append({
                    "id": results["ids"][i],
                    "document": doc,
                    "metadata": results["metadatas"][i] if results["metadatas"] else {},
                })
            items.sort(key=lambda x: x["metadata"].get("cycle_id", 0), reverse=True)
        
        return items[:n_results]
    
    def get_uncertainty_artifacts(self, n_results: int = 5) -> list[dict[str, Any]]:
        """Get artifacts that contain uncertainties"""
        return self.query_by_metadata(
            collection="artifacts",
            where={"has_uncertainty": True},
            n_results=n_results,
        )
    
    def get_profile_change_artifacts(self, n_results: int = 5) -> list[dict[str, Any]]:
        """Get artifacts where profile changes were proposed"""
        return self.query_by_metadata(
            collection="artifacts",
            where={"has_profile_update": True},
            n_results=n_results,
        )
    
    def count(self, collection: str = "artifacts") -> int:
        """Count items in a collection"""
        coll = {
            "artifacts": self.artifacts_collection,
            "concepts": self.concepts_collection,
            "world": self.world_summaries_collection,
        }.get(collection, self.artifacts_collection)
        return coll.count()
    
    def persist(self) -> None:
        """Persist the database to disk"""
        self.client.persist()

