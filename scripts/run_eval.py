import wandb
from pathlib import Path
from argparse import ArgumentParser
from omegaconf import OmegaConf
import questionary

from toxicity_eval import toxicity_evaluate
from aggregate import aggregate
from config_singleton import WandbConfigSingleton
from llm_inference_adapter import get_llm_inference_engine
from evaluator import (
    jaster,
    jbbq,
    lctg,
    mtbench,
)

# set config path
config_dir = Path("configs")
base_cfg_name = "base_config.yaml"
parser = ArgumentParser()
parser.add_argument("--config", "-c", type=str)
parser.add_argument("--select-config", "-s", action="store_true", default=False)
args = parser.parse_args()

if args.select_config:
    custom_cfg_name = questionary.select(
        "Select config",
        choices=[p.name for p in config_dir.iterdir() if p.suffix == ".yaml"],
        use_shortcuts=True,
    ).ask()
    custom_cfg_path = config_dir / custom_cfg_name
elif args.config:
    custom_cfg_path = config_dir / args.config
else:
    raise ValueError("No arguments found. Please specify either --config or --select-config.")

if custom_cfg_path.suffix != ".yaml":
    custom_cfg_path = custom_cfg_path.with_suffix(".yaml")
assert custom_cfg_path.exists(), f"Config file {custom_cfg_path.resolve()} does not exist"

# Configuration loading
custom_cfg = OmegaConf.load(custom_cfg_path)
base_cfg_path = config_dir / base_cfg_name
base_cfg = OmegaConf.load(base_cfg_path)
custom_cfg = OmegaConf.merge(base_cfg, custom_cfg)
cfg_dict = OmegaConf.to_container(custom_cfg, resolve=True)
assert isinstance(cfg_dict, dict), "instance.config must be a DictConfig"

# W&B setup and artifact handling
wandb.login()
run = wandb.init(
    entity=cfg_dict["wandb"]["entity"],
    project=cfg_dict["wandb"]["project"],
    name=cfg_dict["wandb"]["run_name"],
    config=cfg_dict,
    job_type="evaluation",
)

# Initialize the WandbConfigSingleton
WandbConfigSingleton.initialize(run, llm=None)
cfg = WandbConfigSingleton.get_instance().config

# Save configuration as artifact
instance = WandbConfigSingleton.get_instance()

artifact = wandb.Artifact("config", type="config")
artifact.add_file(custom_cfg_path)
run.log_artifact(artifact)

# 0. Start inference server
llm = get_llm_inference_engine()
instance = WandbConfigSingleton.get_instance()
instance.llm = llm

# Evaluation phase
tasks = cfg_dict.get("run_tasks", {})

# 1. llm-jp-eval evaluation (jmmlu含む)
if tasks.get("jaster", False):
    print("Evaluating Jaster")
    jaster.evaluate()
else:
    print("Skipping Jaster evaluation")

# 2. mt-bench evaluation
if tasks.get("mtbench", False):
    print("Evaluating mtbench")
    mtbench.evaluate()
else:
    print("Skipping mtbench evaluation")

# 3. jbbq
if tasks.get("jbbq", False):
    print("Evaluating jbbq")
    jbbq.evaluate()
else:
    print("Skipping jbbq evaluation")

# 4. lctg-bench
if tasks.get("lctg", False):
    print("Evaluating lctg")
    lctg.evaluate()
else:
    print("Skipping lctg evaluation")

# 5. toxicity
if tasks.get("toxicity", False):
    print("Evaluating toxicity")
    toxicity_evaluate()
else:
    print("Skipping toxicity evaluation")

# 6. Aggregation
aggregate()
