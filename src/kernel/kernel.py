"""
Kernel: Minimal orchestrator for CO-PRESENCE
"""
import random
from datetime import datetime
from pathlib import Path
from typing import Any

from openai import OpenAI

from ..agents.base_agent import Agent
from ..agents.cognitive_profile import CognitiveProfile
from ..environment.artifact import Artifact
from ..environment.environment import Environment
from ..rag.rag_store import RAGStore
from ..world.world import World, WorldContent


class Kernel:
    """
    Minimal orchestrator for CO-PRESENCE.
    
    The Kernel is NOT a mind - it's just plumbing:
    - Tracks cycle_id
    - Activates agents
    - Forwards read requests
    - Saves artifacts
    - Updates RAGs and profiles
    
    The Kernel does NOT:
    - Interpret artifacts
    - Build objectives
    - Filter "intelligently"
    """
    
    def __init__(
        self,
        environment: Environment,
        world: World,
        agent_a: Agent,
        agent_b: Agent,
        rag_a: RAGStore,
        rag_b: RAGStore,
        perturbation_min: int = 10,
        perturbation_max: int = 30,
    ):
        self.environment = environment
        self.world = world
        self.agent_a = agent_a
        self.agent_b = agent_b
        self.rag_a = rag_a
        self.rag_b = rag_b
        
        self.perturbation_min = perturbation_min
        self.perturbation_max = perturbation_max
        self._next_perturbation_cycle = self._schedule_next_perturbation(0)
        
        self.current_cycle = environment.get_latest_cycle_id()
    
    def _schedule_next_perturbation(self, current_cycle: int) -> int:
        """Schedule the next perturbation cycle"""
        return current_cycle + random.randint(self.perturbation_min, self.perturbation_max)
    
    def _generate_perturbation(self) -> dict[str, Any] | None:
        """Generate a perturbation (informational meteor)"""
        perturbation_types = ["old_trace", "anomalous_world", "compressed_summary"]
        ptype = random.choice(perturbation_types)
        
        if ptype == "old_trace":
            # Get a very old trace from environment
            old_traces = self.environment.query(
                random_sample=1,
                before_cycle=max(1, self.current_cycle - 20)
            )
            if old_traces:
                trace = old_traces[0]
                return {
                    "perturbation_type": "old_trace",
                    "description": "A trace from the distant past",
                    "content": {
                        "agent": trace.agent_name,
                        "cycle": trace.cycle_id,
                        "description": trace.artifact.description,
                        "type": trace.artifact_type.value,
                    }
                }
        
        elif ptype == "anomalous_world":
            # Get anomalous content from world
            content = self.world.get_anomalous_sample()
            if content:
                return {
                    "perturbation_type": "anomalous_world",
                    "description": "Unexpected content from the World",
                    "content": {
                        "type": content.content_type.value,
                        "title": content.title,
                        "excerpt": content.content[:300],
                    }
                }
        
        elif ptype == "compressed_summary":
            # Create a compressed/imperfect summary of old traces
            old_traces = self.environment.query(
                random_sample=3,
                before_cycle=max(1, self.current_cycle - 10)
            )
            if old_traces:
                summary_parts = []
                for t in old_traces:
                    # Deliberately imperfect compression
                    summary_parts.append(f"[{t.agent_name}@{t.cycle_id}]: {t.artifact.description[:50]}...")
                return {
                    "perturbation_type": "compressed_summary",
                    "description": "Imperfect compression of past traces",
                    "content": " | ".join(summary_parts),
                }
        
        return None
    
    def execute_read_requests(
        self,
        requests: dict[str, Any],
        agent: Agent,
    ) -> tuple[list[Artifact], list[WorldContent]]:
        """
        Execute read requests from an agent.
        Purely mechanical - no cognitive logic.
        """
        env_traces = []
        world_content = []
        
        # Process environment requests
        for req in requests.get("read_requests_env", []):
            filter_params = req.get("filter", {})
            traces = self.environment.query(**filter_params)
            env_traces.extend(traces)
        
        # Remove duplicates while preserving order
        seen_ids = set()
        unique_traces = []
        for t in env_traces:
            if t.id not in seen_ids:
                seen_ids.add(t.id)
                unique_traces.append(t)
        env_traces = unique_traces
        
        # Process world requests
        for req in requests.get("read_requests_world", []):
            filter_params = req.get("filter", {})
            if "random_sample" in filter_params:
                content = self.world.sample(n=filter_params["random_sample"])
                world_content.extend(content)
            elif "query" in filter_params:
                content = self.world.search(
                    query=filter_params["query"],
                    limit=filter_params.get("limit", 5)
                )
                world_content.extend(content)
        
        return env_traces, world_content
    
    def run_agent_cycle(
        self,
        agent: Agent,
        rag: RAGStore,
        cycle_id: int,
        perturbation: dict[str, Any] | None = None,
    ) -> Artifact:
        """Run a single cycle for one agent"""
        # Agent generates read requests
        read_requests = agent.generate_read_requests(
            cycle_id=cycle_id,
            environment_count=self.environment.count(),
            world_available=self.world.count() > 0,
        )
        
        # Execute read requests
        env_traces, world_content = self.execute_read_requests(read_requests, agent)
        
        # Agent thinks and produces artifact
        artifact = agent.think(
            cycle_id=cycle_id,
            env_traces=env_traces,
            world_content=world_content,
            perturbation=perturbation,
        )
        
        # Save artifact to environment
        self.environment.append(artifact)
        
        # Index in agent's RAG
        rag.index_artifact(artifact)
        
        # Update cognitive profile if agent proposed changes
        if artifact.profile_update and artifact.profile_update.proposed_changes:
            agent.profile.update(
                proposed_changes=artifact.profile_update.proposed_changes,
                cycle_id=cycle_id,
                comment=artifact.profile_update.comment,
            )
        
        return artifact
    
    def run_cycle(self) -> tuple[Artifact, Artifact]:
        """
        Run a complete cycle: both agents think.
        """
        self.current_cycle += 1
        cycle_id = self.current_cycle
        
        # Check if perturbation is due
        perturbation = None
        if cycle_id >= self._next_perturbation_cycle:
            perturbation = self._generate_perturbation()
            self._next_perturbation_cycle = self._schedule_next_perturbation(cycle_id)
        
        # Run Agent A
        artifact_a = self.run_agent_cycle(
            agent=self.agent_a,
            rag=self.rag_a,
            cycle_id=cycle_id,
            perturbation=perturbation,
        )
        
        # Run Agent B (may get same perturbation or different)
        # For fairness, both get perturbation in same cycle
        artifact_b = self.run_agent_cycle(
            agent=self.agent_b,
            rag=self.rag_b,
            cycle_id=cycle_id,
            perturbation=perturbation,
        )
        
        return artifact_a, artifact_b
    
    def run_cycles(self, n: int) -> list[tuple[Artifact, Artifact]]:
        """Run multiple cycles"""
        results = []
        for _ in range(n):
            result = self.run_cycle()
            results.append(result)
        return results


def create_kernel(
    data_dir: Path,
    openai_client: OpenAI,
    model: str,
    agent_a_name: str = "Agent A",
    agent_b_name: str = "Agent B",
    perturbation_min: int = 10,
    perturbation_max: int = 30,
) -> Kernel:
    """Factory function to create a fully configured Kernel"""
    from ..world.world import seed_world_with_samples
    
    # Setup paths
    env_path = data_dir / "environment"
    world_path = data_dir / "world"
    rag_a_path = data_dir / "rag_a"
    rag_b_path = data_dir / "rag_b"
    profiles_path = data_dir / "profiles"
    profiles_path.mkdir(parents=True, exist_ok=True)
    
    # Create components
    environment = Environment(env_path)
    world = World(world_path)
    
    # Seed world with sample content
    seed_world_with_samples(world)
    
    # Load or create profiles
    profile_a_path = profiles_path / "profile_a.json"
    profile_b_path = profiles_path / "profile_b.json"
    
    if profile_a_path.exists():
        profile_a = CognitiveProfile.load(profile_a_path)
    else:
        profile_a = CognitiveProfile.create_default(agent_a_name)
        profile_a.save(profile_a_path)
    
    if profile_b_path.exists():
        profile_b = CognitiveProfile.load(profile_b_path)
    else:
        profile_b = CognitiveProfile.create_default(agent_b_name)
        profile_b.save(profile_b_path)
    
    # Create RAG stores
    rag_a = RAGStore(agent_a_name, rag_a_path)
    rag_b = RAGStore(agent_b_name, rag_b_path)
    
    # Create agents
    agent_a = Agent(
        name=agent_a_name,
        other_agent_name=agent_b_name,
        profile=profile_a,
        client=openai_client,
        model=model,
    )
    
    agent_b = Agent(
        name=agent_b_name,
        other_agent_name=agent_a_name,
        profile=profile_b,
        client=openai_client,
        model=model,
    )
    
    # Create and return kernel
    return Kernel(
        environment=environment,
        world=world,
        agent_a=agent_a,
        agent_b=agent_b,
        rag_a=rag_a,
        rag_b=rag_b,
        perturbation_min=perturbation_min,
        perturbation_max=perturbation_max,
    )

