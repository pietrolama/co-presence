#!/usr/bin/env python3
"""
CO-PRESENCE Runner
Launch the cognitive co-presence experiment.
"""
import sys
from pathlib import Path

# Add the project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.main import main

if __name__ == "__main__":
    main()

