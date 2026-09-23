"""Run the shared model suite against one immutable population and global development seal."""

from pricesanity.benchmark.execution import BenchmarkExecutor
from pricesanity.benchmark.protocol import BenchmarkTrack
from pricesanity.benchmark.registry import list_model_families


def execute_suite(executor: BenchmarkExecutor, *, acknowledge_scaling_risk: bool = False) -> list:
    """Finish both tracks' development before permitting any selected final fit."""
    models = [family.name for family in list_model_families() if family.name in executor.scope["models"]]
    tracks = [track for track in BenchmarkTrack if track.value in executor.scope["tracks"]]
    if not (executor.paths.root / "development_frozen.json").exists():
        for track in tracks:
            for model in models:
                executor.tune_model(model, track=track, acknowledge_scaling_risk=acknowledge_scaling_risk)
                executor.run_learning_curve(model, track=track)
    # Revalidate an existing seal on resume; it does not authorize new development work.
    executor.freeze_development()
    paths = []
    for track in tracks:
        for model in models:
            paths.extend(executor.run_final(model, track=track, confirm_final_holdout=True))
    return paths
