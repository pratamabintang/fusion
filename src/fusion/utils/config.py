"""Configuration loading and persistence utilities."""

from pathlib import Path
import yaml


def load_config(config_path: str = None, overrides: dict = None) -> dict:
    """Load and merge configuration from a YAML file, default base config, and CLI overrides."""
    if config_path is None:
        config_path = str(Path(__file__).parent.parent.parent.parent / "configs" / "default.yaml")
        if not Path(config_path).exists():
            config_path = str(Path("D:/fusion/configs/default.yaml"))

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    # Inherit base defaults from default.yaml if loading a preset variant
    default_path = Path("D:/fusion/configs/default.yaml")
    if Path(config_path) != default_path and default_path.exists():
        with open(default_path, "r", encoding="utf-8") as f:
            default_config = yaml.safe_load(f) or {}
        default_config.update(config)
        config = default_config

    if overrides:
        config.update(overrides)

    return config


def save_config(config: dict, save_path: str):
    """Save configuration dictionary to a YAML file."""
    with open(save_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f)
