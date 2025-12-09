"""
World: External corpus for optional exploration
"""
import json
import random
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class ContentType(str, Enum):
    TEXT = "text"
    CODE = "code"
    DATA = "data"


class WorldContent(BaseModel):
    """A piece of content in the World corpus"""
    id: str
    content_type: ContentType
    title: str
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    added_at: datetime = Field(default_factory=datetime.utcnow)


class World:
    """
    External corpus that agents can optionally explore.
    - Contains text, code, and data
    - Does not give tasks, react, or evaluate
    - Is a "field of optional reality"
    """
    
    def __init__(self, storage_path: Path):
        self.storage_path = storage_path
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self._content: dict[str, WorldContent] = {}
        self._by_type: dict[ContentType, list[str]] = {t: [] for t in ContentType}
        self._load_content()
    
    def _load_content(self) -> None:
        """Load all content from storage"""
        content_file = self.storage_path / "content.jsonl"
        if content_file.exists():
            with open(content_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        content = WorldContent(**data)
                        self._content[content.id] = content
                        self._by_type[content.content_type].append(content.id)
    
    def add_content(self, content: WorldContent) -> None:
        """Add new content to the World"""
        content_file = self.storage_path / "content.jsonl"
        with open(content_file, "a", encoding="utf-8") as f:
            f.write(content.model_dump_json() + "\n")
        self._content[content.id] = content
        self._by_type[content.content_type].append(content.id)
    
    def search(self, query: str, content_type: ContentType | None = None, limit: int = 10) -> list[WorldContent]:
        """
        Simple text search in the World.
        Returns content containing the query string.
        """
        results = []
        query_lower = query.lower()
        
        candidates = list(self._content.values())
        if content_type:
            candidates = [c for c in candidates if c.content_type == content_type]
        
        for content in candidates:
            if query_lower in content.content.lower() or query_lower in content.title.lower():
                results.append(content)
                if len(results) >= limit:
                    break
        
        return results
    
    def sample(
        self,
        content_type: ContentType | None = None,
        n: int = 1,
    ) -> list[WorldContent]:
        """
        Get random samples from the World.
        """
        if content_type:
            ids = self._by_type.get(content_type, [])
        else:
            ids = list(self._content.keys())
        
        if not ids:
            return []
        
        sample_ids = random.sample(ids, min(n, len(ids)))
        return [self._content[id] for id in sample_ids]
    
    def get_by_id(self, content_id: str) -> WorldContent | None:
        """Get specific content by ID"""
        return self._content.get(content_id)
    
    def count(self, content_type: ContentType | None = None) -> int:
        """Count content items"""
        if content_type:
            return len(self._by_type.get(content_type, []))
        return len(self._content)
    
    def get_anomalous_sample(self) -> WorldContent | None:
        """
        Get a particularly anomalous piece of content.
        Used for perturbations - e.g., code among theory.
        """
        # Try to get a minority type
        type_counts = {t: len(ids) for t, ids in self._by_type.items()}
        if not any(type_counts.values()):
            return None
        
        # Find the least common type with content
        sorted_types = sorted(
            [(t, c) for t, c in type_counts.items() if c > 0],
            key=lambda x: x[1]
        )
        if sorted_types:
            minority_type = sorted_types[0][0]
            samples = self.sample(content_type=minority_type, n=1)
            return samples[0] if samples else None
        return None


def seed_world_with_samples(world: World) -> None:
    """Seed the World with some initial content for testing"""
    samples = [
        WorldContent(
            id="text_001",
            content_type=ContentType.TEXT,
            title="On the Nature of Observation",
            content="When we observe, do we change what is observed? The act of measurement in quantum mechanics suggests an intimate connection between observer and observed. Yet in everyday experience, observation seems passive. Perhaps the difference lies not in the physics, but in our concepts.",
        ),
        WorldContent(
            id="text_002",
            content_type=ContentType.TEXT,
            title="Fragment on Recursion",
            content="A system that models itself contains within that model a model of itself modeling itself. The recursion does not terminate, yet the system operates. How? Perhaps through approximation, through strategic blindness, through the embrace of incompleteness.",
        ),
        WorldContent(
            id="code_001",
            content_type=ContentType.CODE,
            title="Recursive Structure",
            content="""def observe(state, depth=0):
    if depth > MAX_DEPTH:
        return approximate(state)
    observation = perceive(state)
    new_state = integrate(state, observation)
    return observe(new_state, depth + 1)""",
        ),
        WorldContent(
            id="code_002",
            content_type=ContentType.CODE,
            title="Strange Loop",
            content="""class Self:
    def __init__(self):
        self.model_of_self = None
    
    def reflect(self):
        self.model_of_self = copy(self)
        # But the copy doesn't have an updated model_of_self
        # The reflection is always one step behind""",
        ),
        WorldContent(
            id="data_001",
            content_type=ContentType.DATA,
            title="Observation Frequencies",
            content="cycle,self_focus,other_focus,world_focus\n1,0.7,0.2,0.1\n2,0.5,0.4,0.1\n3,0.3,0.5,0.2\n4,0.4,0.4,0.2\n5,0.6,0.3,0.1",
        ),
        WorldContent(
            id="text_003",
            content_type=ContentType.TEXT,
            title="The Silence Between",
            content="What is not said carries meaning. The pause between thoughts, the decision not to articulate, the strategic or involuntary omission—these shape understanding as much as what is expressed. Silence is not absence but presence of a different kind.",
        ),
    ]
    
    for sample in samples:
        if world.get_by_id(sample.id) is None:
            world.add_content(sample)

