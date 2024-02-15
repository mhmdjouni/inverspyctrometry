from pathlib import Path
from typing import Optional

from src.interface.configuration import load_config


def experiment_dir_convention(
        dir_type: str,
        experiment_id: int,
        custom_dir: Optional[Path] = None,
) -> Path:
    """If no custom_dir is provided, the default is the reports folder"""
    config = load_config()
    if custom_dir is None:
        custom_dir = config.directory_paths.reports

    if dir_type == "simulation":
        return custom_dir / f"experiment_{experiment_id}" / "simulation"
    elif dir_type == "reconstruction":
        return custom_dir / f"experiment_{experiment_id}" / "reconstruction"
    elif dir_type == "metrics":
        return custom_dir / f"experiment_{experiment_id}" / "metrics"
    elif dir_type == "figures":
        return custom_dir / f"experiment_{experiment_id}" / "figures"
    elif dir_type == "paper_figures":
        db = config.database()
        return custom_dir / "figures" / f"{db.experiments[experiment_id].type}"
    else:
        raise ValueError()


def experiment_subdir_convention(
        dataset_id: int,
        interferometer_id: int = -1,
        noise_level_index: int = -1,
        inversion_protocol_id: int = -1,
):
    config = load_config()
    db = config.database()
    subdir = f"invert_{db.datasets[dataset_id].title}"
    if interferometer_id >= 0:
        subdir = f"{subdir}/{db.interferometers[interferometer_id].title}"
    if noise_level_index >= 0:
        subdir = f"{subdir}/{int(db.noise_levels[noise_level_index])}_db"
    if inversion_protocol_id >= 0:
        subdir = f"{subdir}/{db.inversion_protocols[inversion_protocol_id].title}"
    return subdir
