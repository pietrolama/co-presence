"""
Base Agent implementation for CO-PRESENCE
"""
import json
import re
from typing import Any

from openai import OpenAI

from ..environment.artifact import (
    Artifact,
    ArtifactContent,
    ArtifactType,
    MetaCognition,
    ProfileUpdate,
    Step,
)
from .cognitive_profile import CognitiveProfile
from .system_prompt import get_system_prompt


class Agent:
    """
    A CO-PRESENCE agent: wrapper around LLM with cognitive profile.
    """
    
    def __init__(
        self,
        name: str,
        other_agent_name: str,
        profile: CognitiveProfile,
        client: OpenAI,
        model: str,
    ):
        self.name = name
        self.other_agent_name = other_agent_name
        self.profile = profile
        self.client = client
        self.model = model
        self.system_prompt = get_system_prompt(name, other_agent_name)
    
    def generate_read_requests(
        self,
        cycle_id: int,
        environment_count: int,
        world_available: bool,
    ) -> dict[str, Any]:
        """
        Agent autonomously decides what to read.
        Based on cognitive profile preferences.
        """
        p = self.profile.preferences
        
        requests = {
            "read_requests_env": [],
            "read_requests_world": [],
        }
        
        # Self focus: recent own traces
        if p.self_focus >= 0.3:
            limit = max(1, int(5 * p.self_focus))
            requests["read_requests_env"].append({
                "label": "recent_self",
                "filter": {"agent_name": self.name, "limit": limit, "order": "desc"}
            })
        
        # Other focus: recent other's traces
        if p.other_focus >= 0.3:
            limit = max(1, int(5 * p.other_focus))
            requests["read_requests_env"].append({
                "label": "recent_other",
                "filter": {"agent_name": self.other_agent_name, "limit": limit, "order": "desc"}
            })
        
        # Historical sampling based on abstraction level
        if p.abstraction_level.value == "high" and environment_count > 10:
            requests["read_requests_env"].append({
                "label": "historical_sample",
                "filter": {"random_sample": 3}
            })
        
        # World focus
        if world_available and p.world_focus >= 0.3:
            n_samples = max(1, int(3 * p.world_focus))
            requests["read_requests_world"].append({
                "label": "random_world",
                "filter": {"random_sample": n_samples}
            })
        
        return requests
    
    def think(
        self,
        cycle_id: int,
        env_traces: list[Artifact],
        world_content: list[Any],
        perturbation: dict[str, Any] | None = None,
    ) -> Artifact:
        """
        Core thinking process: generate an artifact from context.
        """
        # Build context message
        context_parts = []
        
        # Add cycle info
        context_parts.append(f"[CYCLE: {cycle_id}]")
        
        # Add cognitive profile
        context_parts.append(self.profile.to_prompt_context())
        
        # Add environment traces
        if env_traces:
            context_parts.append("\n[ENVIRONMENT TRACES]")
            for trace in env_traces:
                context_parts.append(self._format_trace(trace))
        else:
            context_parts.append("\n[ENVIRONMENT TRACES]\nNo traces available yet.")
        
        # Add world content
        if world_content:
            context_parts.append("\n[WORLD CONTENT]")
            for content in world_content:
                context_parts.append(self._format_world_content(content))
        
        # Add perturbation if present
        if perturbation:
            context_parts.append("\n[PERTURBATION - UNEXPECTED INPUT]")
            context_parts.append(json.dumps(perturbation, indent=2, ensure_ascii=False))
        
        user_message = "\n".join(context_parts)
        
        # Call LLM
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=0.8,
            max_tokens=2000,
        )
        
        raw_output = response.choices[0].message.content
        
        # Parse and validate output
        artifact = self._parse_output(raw_output, cycle_id)
        
        return artifact
    
    def _format_trace(self, trace: Artifact) -> str:
        """Format an artifact trace for context"""
        return f"""
--- Trace from {trace.agent_name} (cycle {trace.cycle_id}) ---
Type: {trace.artifact_type.value}
Description: {trace.artifact.description}
Steps: {json.dumps([s.model_dump() for s in trace.artifact.steps], ensure_ascii=False)}
Meta-cognition: {trace.artifact.meta_cognition.model_dump()}
Silence: {trace.silence_flag}
"""
    
    def _format_world_content(self, content: Any) -> str:
        """Format world content for context"""
        if hasattr(content, 'model_dump'):
            return f"""
--- World Content: {content.title} ({content.content_type.value}) ---
{content.content}
"""
        return str(content)
    
    def _parse_output(self, raw_output: str, cycle_id: int) -> Artifact:
        """Parse LLM output into an Artifact"""
        # Try to extract JSON from the output
        try:
            # Clean potential markdown code blocks
            cleaned = raw_output.strip()
            if cleaned.startswith("```"):
                # Remove markdown code fences
                cleaned = re.sub(r'^```(?:json)?\s*', '', cleaned)
                cleaned = re.sub(r'\s*```$', '', cleaned)
            
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            # Fallback: create error artifact
            return self._create_error_artifact(cycle_id, raw_output)
        
        # Build artifact from parsed data
        try:
            steps = [Step(**s) for s in data.get("artifact", {}).get("steps", [])]
            meta_cog_data = data.get("artifact", {}).get("meta_cognition", {})
            meta_cognition = MetaCognition(
                self_observation=meta_cog_data.get("self_observation", "Unable to observe self"),
                influence_of_other_agent=meta_cog_data.get("influence_of_other_agent", "Unknown"),
                uncertainties=meta_cog_data.get("uncertainties", "Cannot determine"),
            )
            
            artifact_content = ArtifactContent(
                description=data.get("artifact", {}).get("description", "No description"),
                steps=steps,
                meta_cognition=meta_cognition,
            )
            
            profile_update = None
            if data.get("profile_update"):
                profile_update = ProfileUpdate(
                    proposed_changes=data["profile_update"].get("proposed_changes", {}),
                    comment=data["profile_update"].get("comment", ""),
                )
            
            artifact_type_str = data.get("artifact_type", "partial_theory")
            try:
                artifact_type = ArtifactType(artifact_type_str)
            except ValueError:
                artifact_type = ArtifactType.PARTIAL_THEORY
            
            return Artifact(
                agent_name=self.name,
                cycle_id=cycle_id,
                artifact_type=artifact_type,
                artifact=artifact_content,
                profile_update=profile_update,
                profile_snapshot=self.profile.preferences.model_dump(),
                silence_flag=data.get("silence_flag", False),
            )
        except Exception as e:
            return self._create_error_artifact(cycle_id, f"Parse error: {e}\n\nRaw: {raw_output}")
    
    def _create_error_artifact(self, cycle_id: int, error_info: str) -> Artifact:
        """Create an artifact for parsing errors"""
        return Artifact(
            agent_name=self.name,
            cycle_id=cycle_id,
            artifact_type=ArtifactType.PARTIAL_THEORY,
            artifact=ArtifactContent(
                description="Error parsing agent output",
                steps=[Step(label="error", content=error_info[:500])],
                meta_cognition=MetaCognition(
                    self_observation="Output parsing failed",
                    influence_of_other_agent="N/A",
                    uncertainties="Cannot determine due to error",
                ),
            ),
            profile_snapshot=self.profile.preferences.model_dump(),
            silence_flag=False,
        )

