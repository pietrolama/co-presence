"""
CO-PRESENCE: Cognitive Co-Presence Experiment

Main entry point for running the experiment.
"""
import argparse
import json
from datetime import datetime
from pathlib import Path

from openai import OpenAI
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    DATA_DIR,
    LLM_MODEL,
    OPENAI_API_KEY,
    OPENAI_BASE_URL,
    PERTURBATION_MAX_CYCLES,
    PERTURBATION_MIN_CYCLES,
    RUNS_DIR,
    AGENT_A_NAME,
    AGENT_B_NAME,
)
from .kernel.kernel import create_kernel
from .observer.observer import Observer

console = Console()


def create_run_directory() -> Path:
    """Create a timestamped directory for this run [[memory:4243901]]"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = RUNS_DIR / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def display_artifact_summary(artifact, agent_label: str) -> None:
    """Display a summary of an artifact"""
    table = Table(show_header=False, box=None)
    table.add_column("Field", style="cyan")
    table.add_column("Value")
    
    table.add_row("Type", artifact.artifact_type.value)
    table.add_row("Description", artifact.artifact.description[:100] + "..." if len(artifact.artifact.description) > 100 else artifact.artifact.description)
    table.add_row("Steps", str(len(artifact.artifact.steps)))
    table.add_row("Silent", "Yes" if artifact.silence_flag else "No")
    
    if artifact.profile_update and artifact.profile_update.proposed_changes:
        table.add_row("Profile Update", json.dumps(artifact.profile_update.proposed_changes))
    
    console.print(Panel(table, title=f"[bold]{agent_label}[/bold] - Cycle {artifact.cycle_id}"))


def run_experiment(
    num_cycles: int,
    model: str = LLM_MODEL,
    save_each_cycle: bool = True,
) -> None:
    """Run the CO-PRESENCE experiment"""
    
    # Create run directory
    run_dir = create_run_directory()
    console.print(f"[green]Run directory:[/green] {run_dir}")
    
    # Initialize OpenAI client
    client = OpenAI(
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_BASE_URL,
    )
    
    # Create kernel with all components
    console.print("[yellow]Initializing kernel...[/yellow]")
    kernel = create_kernel(
        data_dir=DATA_DIR,
        openai_client=client,
        model=model,
        agent_a_name=AGENT_A_NAME,
        agent_b_name=AGENT_B_NAME,
        perturbation_min=PERTURBATION_MIN_CYCLES,
        perturbation_max=PERTURBATION_MAX_CYCLES,
    )
    
    # Initialize observer
    observer = Observer(run_dir / "metrics")
    
    console.print(f"[green]Starting experiment with {num_cycles} cycles[/green]")
    console.print(f"[dim]Model: {model}[/dim]")
    console.print()
    
    # Run cycles
    for i in range(num_cycles):
        cycle_num = kernel.current_cycle + 1
        console.print(f"[bold blue]━━━ Cycle {cycle_num} ━━━[/bold blue]")
        
        try:
            artifact_a, artifact_b = kernel.run_cycle()
            
            # Display summaries
            display_artifact_summary(artifact_a, AGENT_A_NAME)
            display_artifact_summary(artifact_b, AGENT_B_NAME)
            
            # Compute and log metrics
            cycle_metrics = observer.compute_cycle_metrics(artifact_a, artifact_b)
            observer.log_cycle(cycle_metrics)
            
            # Save artifacts if requested
            if save_each_cycle:
                cycle_file = run_dir / "artifacts" / f"cycle_{cycle_num:04d}.json"
                cycle_file.parent.mkdir(parents=True, exist_ok=True)
                with open(cycle_file, "w", encoding="utf-8") as f:
                    json.dump({
                        "cycle_id": cycle_num,
                        "agent_a": artifact_a.model_dump(mode="json"),
                        "agent_b": artifact_b.model_dump(mode="json"),
                    }, f, indent=2, ensure_ascii=False)
            
            console.print()
            
        except Exception as e:
            console.print(f"[red]Error in cycle {cycle_num}: {e}[/red]")
            raise
    
    # Generate final summary
    console.print("[yellow]Generating summary...[/yellow]")
    summary = observer.compute_summary(kernel.environment)
    
    # Save profiles
    profiles_dir = run_dir / "final_profiles"
    profiles_dir.mkdir(parents=True, exist_ok=True)
    kernel.agent_a.profile.save(profiles_dir / "agent_a_profile.json")
    kernel.agent_b.profile.save(profiles_dir / "agent_b_profile.json")
    
    # Display summary
    console.print()
    console.print(Panel(
        json.dumps(summary, indent=2, default=str),
        title="[bold green]Experiment Summary[/bold green]",
    ))
    
    console.print(f"\n[green]Experiment complete. Results saved to:[/green] {run_dir}")


def main():
    parser = argparse.ArgumentParser(description="CO-PRESENCE: Cognitive Co-Presence Experiment")
    parser.add_argument(
        "-n", "--cycles",
        type=int,
        default=5,
        help="Number of cycles to run (default: 5)"
    )
    parser.add_argument(
        "-m", "--model",
        type=str,
        default=LLM_MODEL,
        help=f"LLM model to use (default: {LLM_MODEL})"
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save individual cycle artifacts"
    )
    
    args = parser.parse_args()
    
    if not OPENAI_API_KEY:
        console.print("[red]Error: OPENAI_API_KEY not set. Create a .env file with your API key.[/red]")
        return
    
    run_experiment(
        num_cycles=args.cycles,
        model=args.model,
        save_each_cycle=not args.no_save,
    )


if __name__ == "__main__":
    main()

