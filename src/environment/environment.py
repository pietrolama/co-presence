"""
Environment: Shared append-only artifact collection
"""
import json
import random
from datetime import datetime
from pathlib import Path
from typing import Literal

from .artifact import Artifact


class Environment:
    """
    Shared environment for CO-PRESENCE agents.
    - Append-only collection of artifacts
    - No filtering, evaluation, or weighting
    - Raw query execution only
    """
    
    def __init__(self, storage_path: Path):
        self.storage_path = storage_path
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.artifacts_file = self.storage_path / "artifacts.jsonl"
        self._cache: list[Artifact] = []
        self._load_artifacts()
    
    def _load_artifacts(self) -> None:
        """Load all artifacts from storage"""
        self._cache = []
        if self.artifacts_file.exists():
            with open(self.artifacts_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        self._cache.append(Artifact(**data))
    
    def append(self, artifact: Artifact) -> None:
        """Append a new artifact (immutable after creation)"""
        with open(self.artifacts_file, "a", encoding="utf-8") as f:
            f.write(artifact.model_dump_json() + "\n")
        self._cache.append(artifact)
    
    def query(
        self,
        agent_name: str | None = None,
        limit: int | None = None,
        order: Literal["asc", "desc"] = "desc",
        artifact_type: str | None = None,
        cycle_range: tuple[int, int] | None = None,
        random_sample: int | None = None,
        has_uncertainty: bool | None = None,
        has_profile_update: bool | None = None,
        before_cycle: int | None = None,
        after_cycle: int | None = None,
    ) -> list[Artifact]:
        """
        Query artifacts with various filters.
        The Environment executes queries "stupidly" - no cognitive logic.
        """
        results = list(self._cache)
        
        # Filter by agent
        if agent_name:
            results = [a for a in results if a.agent_name == agent_name]
        
        # Filter by artifact type
        if artifact_type:
            results = [a for a in results if a.artifact_type.value == artifact_type]
        
        # Filter by cycle range
        if cycle_range:
            min_cycle, max_cycle = cycle_range
            results = [a for a in results if min_cycle <= a.cycle_id <= max_cycle]
        
        # Filter before/after cycle
        if before_cycle is not None:
            results = [a for a in results if a.cycle_id < before_cycle]
        if after_cycle is not None:
            results = [a for a in results if a.cycle_id > after_cycle]
        
        # Filter by uncertainty presence
        if has_uncertainty is not None:
            if has_uncertainty:
                results = [a for a in results if a.artifact.meta_cognition.uncertainties.strip()]
            else:
                results = [a for a in results if not a.artifact.meta_cognition.uncertainties.strip()]
        
        # Filter by profile update presence
        if has_profile_update is not None:
            if has_profile_update:
                results = [a for a in results if a.profile_update and a.profile_update.proposed_changes]
            else:
                results = [a for a in results if not (a.profile_update and a.profile_update.proposed_changes)]
        
        # Sort by timestamp/cycle
        results.sort(key=lambda a: (a.cycle_id, a.timestamp), reverse=(order == "desc"))
        
        # Random sampling
        if random_sample is not None and random_sample < len(results):
            results = random.sample(results, random_sample)
        
        # Limit
        if limit is not None:
            results = results[:limit]
        
        return results
    
    def get_all(self) -> list[Artifact]:
        """Get all artifacts"""
        return list(self._cache)
    
    def get_by_cycle(self, cycle_id: int) -> list[Artifact]:
        """Get all artifacts from a specific cycle"""
        return [a for a in self._cache if a.cycle_id == cycle_id]
    
    def get_latest_cycle_id(self) -> int:
        """Get the latest cycle ID"""
        if not self._cache:
            return 0
        return max(a.cycle_id for a in self._cache)
    
    def count(self) -> int:
        """Count total artifacts"""
        return len(self._cache)

