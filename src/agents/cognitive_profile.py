"""
Cognitive Profile: Agent's cognitive preferences and tendencies
"""
import json
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class AbstractionLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ComplexityTarget(str, Enum):
    SIMPLE = "simple"
    COMPLEX = "complex"
    VARIABLE = "variable"


class CognitivePreferences(BaseModel):
    """Cognitive preference parameters"""
    abstraction_level: AbstractionLevel = AbstractionLevel.MEDIUM
    tendency_to_close: float = Field(default=0.5, ge=0.0, le=1.0, description="0=leaves open, 1=always closes")
    self_focus: float = Field(default=0.5, ge=0.0, le=1.0, description="Weight on own traces")
    other_focus: float = Field(default=0.5, ge=0.0, le=1.0, description="Weight on other's traces")
    world_focus: float = Field(default=0.3, ge=0.0, le=1.0, description="Tendency to consult World")
    complexity_target: ComplexityTarget = ComplexityTarget.VARIABLE


class CognitiveProfile(BaseModel):
    """
    Persistent cognitive profile for an agent.
    Describes current cognitive preferences.
    """
    agent_name: str
    preferences: CognitivePreferences = Field(default_factory=CognitivePreferences)
    history: list[dict[str, Any]] = Field(default_factory=list, description="History of profile changes")
    
    def update(self, proposed_changes: dict[str, Any], cycle_id: int, comment: str = "") -> None:
        """
        Update the profile based on agent's proposed changes.
        Records history of changes.
        """
        if not proposed_changes:
            return
        
        # Record the change in history
        change_record = {
            "cycle_id": cycle_id,
            "before": self.preferences.model_dump(),
            "changes": proposed_changes,
            "comment": comment,
        }
        
        # Apply changes to preferences
        current = self.preferences.model_dump()
        for key, value in proposed_changes.items():
            if key in current:
                # Handle enum conversions
                if key == "abstraction_level":
                    value = AbstractionLevel(value)
                elif key == "complexity_target":
                    value = ComplexityTarget(value)
                setattr(self.preferences, key, value)
        
        change_record["after"] = self.preferences.model_dump()
        self.history.append(change_record)
    
    def to_prompt_context(self) -> str:
        """Generate a description for the agent's prompt context"""
        p = self.preferences
        return f"""[COGNITIVE PROFILE - {self.agent_name}]
- Abstraction Level: {p.abstraction_level.value}
- Tendency to Close Reasoning: {p.tendency_to_close:.2f} (0=open, 1=closed)
- Self Focus: {p.self_focus:.2f}
- Other Focus: {p.other_focus:.2f}
- World Focus: {p.world_focus:.2f}
- Complexity Target: {p.complexity_target.value}"""
    
    def save(self, path: Path) -> None:
        """Save profile to file"""
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.model_dump_json(indent=2))
    
    @classmethod
    def load(cls, path: Path) -> "CognitiveProfile":
        """Load profile from file"""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(**data)
    
    @classmethod
    def create_default(cls, agent_name: str) -> "CognitiveProfile":
        """Create a default profile for an agent"""
        return cls(agent_name=agent_name)

