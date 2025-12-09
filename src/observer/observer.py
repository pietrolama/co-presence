"""
Observer: Derived metrics for human analysis
These metrics are NEVER shown to the agents.
"""
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from ..environment.artifact import Artifact
from ..environment.environment import Environment


class AgentMetrics(BaseModel):
    """Metrics for a single agent in a cycle"""
    agent_name: str
    cycle_id: int
    
    # Structural complexity
    step_count: int = 0
    open_end_ratio: float = 0.0  # % of open_end steps vs conclusions
    avg_step_length: float = 0.0
    
    # Silence
    is_silent: bool = False
    
    # Profile dynamics
    profile_changed: bool = False
    profile_changes: dict[str, Any] = Field(default_factory=dict)
    
    # Artifact type
    artifact_type: str = ""


class CycleMetrics(BaseModel):
    """Metrics for a complete cycle"""
    cycle_id: int
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    agent_a: AgentMetrics | None = None
    agent_b: AgentMetrics | None = None
    
    # Cross-agent metrics (optional)
    both_silent: bool = False


class Observer:
    """
    Observer module for CO-PRESENCE.
    Calculates derived metrics for human analysis.
    
    Metrics include:
    - Structural complexity
    - Gaze orientation (self/other/world focus)
    - Profile dynamics
    - Silence patterns
    - (Optional) Semantic divergence
    """
    
    def __init__(self, storage_path: Path):
        self.storage_path = storage_path
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.metrics_file = self.storage_path / "metrics.jsonl"
        self.metrics_csv = self.storage_path / "metrics.csv"
        self.summary_file = self.storage_path / "summary.json"
        self._init_csv()
    
    def compute_agent_metrics(self, artifact: Artifact) -> AgentMetrics:
        """Compute metrics for a single artifact"""
        steps = artifact.artifact.steps
        
        # Count step types
        open_end_count = sum(1 for s in steps if s.label in ["open_end", "open_question"])
        conclusion_count = sum(1 for s in steps if s.label in ["conclusion", "inference"])
        
        # Calculate ratios
        total_closable = open_end_count + conclusion_count
        open_end_ratio = open_end_count / total_closable if total_closable > 0 else 0.0
        
        # Average step length
        step_lengths = [len(s.content) for s in steps]
        avg_step_length = np.mean(step_lengths) if step_lengths else 0.0
        
        # Profile changes
        profile_changed = bool(
            artifact.profile_update and 
            artifact.profile_update.proposed_changes
        )
        profile_changes = (
            artifact.profile_update.proposed_changes 
            if artifact.profile_update else {}
        )
        
        return AgentMetrics(
            agent_name=artifact.agent_name,
            cycle_id=artifact.cycle_id,
            step_count=len(steps),
            open_end_ratio=open_end_ratio,
            avg_step_length=float(avg_step_length),
            is_silent=artifact.silence_flag,
            profile_changed=profile_changed,
            profile_changes=profile_changes,
            artifact_type=artifact.artifact_type.value,
        )
    
    def compute_cycle_metrics(
        self,
        artifact_a: Artifact | None,
        artifact_b: Artifact | None,
    ) -> CycleMetrics:
        """Compute metrics for a complete cycle"""
        cycle_id = artifact_a.cycle_id if artifact_a else (artifact_b.cycle_id if artifact_b else 0)
        
        metrics_a = self.compute_agent_metrics(artifact_a) if artifact_a else None
        metrics_b = self.compute_agent_metrics(artifact_b) if artifact_b else None
        
        both_silent = (
            (metrics_a is not None and metrics_a.is_silent) and
            (metrics_b is not None and metrics_b.is_silent)
        )
        
        return CycleMetrics(
            cycle_id=cycle_id,
            agent_a=metrics_a,
            agent_b=metrics_b,
            both_silent=both_silent,
        )
    
    def _init_csv(self) -> None:
        """Initialize CSV file with headers"""
        if not self.metrics_csv.exists():
            headers = [
                "cycle_id", "timestamp",
                "a_step_count", "a_open_end_ratio", "a_avg_step_length", 
                "a_is_silent", "a_profile_changed", "a_artifact_type",
                "b_step_count", "b_open_end_ratio", "b_avg_step_length",
                "b_is_silent", "b_profile_changed", "b_artifact_type",
                "both_silent"
            ]
            with open(self.metrics_csv, "w", encoding="utf-8") as f:
                f.write(",".join(headers) + "\n")
    
    def log_cycle(self, cycle_metrics: CycleMetrics) -> None:
        """Append cycle metrics to log (JSONL and CSV)"""
        # JSONL
        with open(self.metrics_file, "a", encoding="utf-8") as f:
            f.write(cycle_metrics.model_dump_json() + "\n")
        
        # CSV
        a = cycle_metrics.agent_a
        b = cycle_metrics.agent_b
        row = [
            str(cycle_metrics.cycle_id),
            cycle_metrics.timestamp.isoformat(),
            str(a.step_count if a else 0),
            f"{a.open_end_ratio:.2f}" if a else "0",
            f"{a.avg_step_length:.1f}" if a else "0",
            str(a.is_silent if a else False),
            str(a.profile_changed if a else False),
            a.artifact_type if a else "",
            str(b.step_count if b else 0),
            f"{b.open_end_ratio:.2f}" if b else "0",
            f"{b.avg_step_length:.1f}" if b else "0",
            str(b.is_silent if b else False),
            str(b.profile_changed if b else False),
            b.artifact_type if b else "",
            str(cycle_metrics.both_silent),
        ]
        with open(self.metrics_csv, "a", encoding="utf-8") as f:
            f.write(",".join(row) + "\n")
    
    def compute_summary(self, environment: Environment) -> dict[str, Any]:
        """
        Compute aggregate summary statistics.
        """
        artifacts = environment.get_all()
        
        if not artifacts:
            return {"total_cycles": 0}
        
        # Group by agent
        by_agent: dict[str, list[Artifact]] = {}
        for a in artifacts:
            by_agent.setdefault(a.agent_name, []).append(a)
        
        summary = {
            "total_cycles": environment.get_latest_cycle_id(),
            "total_artifacts": len(artifacts),
            "agents": {},
        }
        
        for agent_name, agent_artifacts in by_agent.items():
            agent_summary = self._compute_agent_summary(agent_name, agent_artifacts)
            summary["agents"][agent_name] = agent_summary
        
        # Silence streak analysis
        summary["silence_analysis"] = self._analyze_silence_patterns(artifacts)
        
        # Profile evolution
        for agent_name, agent_artifacts in by_agent.items():
            summary["agents"][agent_name]["profile_evolution"] = (
                self._analyze_profile_evolution(agent_artifacts)
            )
        
        # Save summary
        with open(self.summary_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, default=str)
        
        return summary
    
    def _compute_agent_summary(
        self,
        agent_name: str,
        artifacts: list[Artifact],
    ) -> dict[str, Any]:
        """Compute summary for a single agent"""
        total = len(artifacts)
        silent_count = sum(1 for a in artifacts if a.silence_flag)
        
        # Structural complexity
        step_counts = [len(a.artifact.steps) for a in artifacts]
        
        # Artifact type distribution
        type_counts: dict[str, int] = {}
        for a in artifacts:
            t = a.artifact_type.value
            type_counts[t] = type_counts.get(t, 0) + 1
        
        # Profile change frequency
        profile_changes = sum(
            1 for a in artifacts 
            if a.profile_update and a.profile_update.proposed_changes
        )
        
        return {
            "total_artifacts": total,
            "silent_count": silent_count,
            "silence_rate": silent_count / total if total > 0 else 0,
            "avg_steps": float(np.mean(step_counts)) if step_counts else 0,
            "max_steps": max(step_counts) if step_counts else 0,
            "min_steps": min(step_counts) if step_counts else 0,
            "artifact_types": type_counts,
            "profile_change_count": profile_changes,
            "profile_change_rate": profile_changes / total if total > 0 else 0,
        }
    
    def _analyze_silence_patterns(self, artifacts: list[Artifact]) -> dict[str, Any]:
        """Analyze silence patterns (consecutive silence streaks)"""
        # Sort by cycle
        sorted_artifacts = sorted(artifacts, key=lambda a: (a.cycle_id, a.agent_name))
        
        # Find silence streaks per agent
        streaks: dict[str, list[int]] = {}
        current_streak: dict[str, int] = {}
        
        for a in sorted_artifacts:
            if a.agent_name not in current_streak:
                current_streak[a.agent_name] = 0
                streaks[a.agent_name] = []
            
            if a.silence_flag:
                current_streak[a.agent_name] += 1
            else:
                if current_streak[a.agent_name] > 0:
                    streaks[a.agent_name].append(current_streak[a.agent_name])
                current_streak[a.agent_name] = 0
        
        # Don't forget ongoing streaks
        for agent, streak in current_streak.items():
            if streak > 0:
                streaks[agent].append(streak)
        
        return {
            agent: {
                "total_streaks": len(s),
                "max_streak": max(s) if s else 0,
                "avg_streak": float(np.mean(s)) if s else 0,
            }
            for agent, s in streaks.items()
        }
    
    def _analyze_profile_evolution(self, artifacts: list[Artifact]) -> list[dict[str, Any]]:
        """Track profile changes over time"""
        changes = []
        for a in sorted(artifacts, key=lambda x: x.cycle_id):
            if a.profile_update and a.profile_update.proposed_changes:
                changes.append({
                    "cycle_id": a.cycle_id,
                    "changes": a.profile_update.proposed_changes,
                    "comment": a.profile_update.comment,
                })
        return changes
    
    def get_gaze_orientation_over_time(
        self,
        environment: Environment,
        agent_name: str,
    ) -> list[dict[str, Any]]:
        """
        Analyze gaze orientation (self/other/world focus) over time.
        Based on profile snapshots stored in artifacts.
        """
        artifacts = environment.query(agent_name=agent_name, order="asc")
        
        orientations = []
        for a in artifacts:
            if a.profile_snapshot:
                orientations.append({
                    "cycle_id": a.cycle_id,
                    "self_focus": a.profile_snapshot.get("self_focus", 0.5),
                    "other_focus": a.profile_snapshot.get("other_focus", 0.5),
                    "world_focus": a.profile_snapshot.get("world_focus", 0.3),
                })
        
        return orientations

