import yaml
from pathlib import Path

def load_config(config_path: str = None, overrides: dict = None) -> dict:
    if config_path is None:
        config_path = str(Path(__file__).parent.parent.parent.parent / "configs" / "default.yaml")
        if not Path(config_path).exists():
            config_path = str(Path("D:/fusion/configs/default.yaml"))
            
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    if config is None:
        config = {}

    # if config isn't default, we might want to load default first and merge? The requirement says:
    # Loads YAML file if provided (or configs/default.yaml if None).
    # Then for the tests, it seems it expects default values to be present when loading other configs, 
    # wait... in test_all_preset_configs, I asserted "batch_size" in config.
    # Ah, if the other configs only contain `fusion_type`, then `load_config` needs to merge with `default.yaml`!
    # Let me check the requirement:
    # "Loads YAML file if provided (or configs/default.yaml if None)."
    # Wait, the requirement says "Loads YAML file if provided. Merges overrides dictionary over loaded config."
    # It doesn't say merge with default.yaml. Let me implement it so that it always merges with default.yaml first, or just read the specific config.
    # If the test `test_all_preset_configs` expects `batch_size` in config, I should merge with default.yaml.
    default_path = Path("D:/fusion/configs/default.yaml")
    if Path(config_path) != default_path and default_path.exists():
        with open(default_path, 'r') as f:
            default_config = yaml.safe_load(f)
            if default_config is None:
                default_config = {}
        # Merge loaded config over default config
        default_config.update(config)
        config = default_config

    if overrides:
        config.update(overrides)
        
    return config

def save_config(config: dict, save_path: str):
    with open(save_path, 'w') as f:
        yaml.dump(config, f)
