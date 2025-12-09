"""
Artifact models for CO-PRESENCE Environment
"""
from datetime import datetime
from enum import Enum
from typing import Any
from pydantic import BaseModel, Field


class ArtifactType(str, Enum):
    HYPOTHESIS_CHAIN = "hypothesis_chain"
    CLASSIFICATION = "classification"
    PARTIAL_THEORY = "partial_theory"
    COMPARISON = "comparison"
    OPEN_QUESTION = "open_question"
    SILENCE = "silence"


class Step(BaseModel):
    """A single step in the reasoning chain"""
    label: str = Field(..., description="Step type: assumption, inference, counterexample, observation, open_end, conclusion")
    content: str = Field(..., description="The thought content of this step")


class MetaCognition(BaseModel):
    """Meta-cognitive reflection section"""
    self_observation: str = Field(..., description="Observations about current thinking style")
    influence_of_other_agent: str = Field(..., description="How the other agent's traces influence thinking")
    uncertainties: str = Field(..., description="What cannot be determined or interpreted")


class ArtifactContent(BaseModel):
    """The main content structure of an artifact"""
    description: str = Field(..., description="Brief conceptual description of the trace")
    steps: list[Step] = Field(default_factory=list, description="Reasoning steps")
    meta_cognition: MetaCognition


class ProfileUpdate(BaseModel):
    """Proposed changes to cognitive profile"""
    proposed_changes: dict[str, Any] = Field(default_factory=dict)
    comment: str = Field(default="", description="Reason for proposed changes")


class Artifact(BaseModel):
    """
    A structured thought trace produced by an agent.
    Append-only, immutable once created.
    """
    agent_name: str
    cycle_id: int
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    artifact_type: ArtifactType
    artifact: ArtifactContent
    profile_update: ProfileUpdate | None = None
    profile_snapshot: dict[str, Any] | None = None
    silence_flag: bool = False
    
    # Internal ID for storage
    id: str | None = None
    
    def model_post_init(self, __context: Any) -> None:
        if self.id is None:
            self.id = f"{self.agent_name.replace(' ', '_').lower()}_{self.cycle_id}_{self.timestamp.strftime('%Y%m%d%H%M%S%f')}"

