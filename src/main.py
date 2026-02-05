import jax
from wrappers import LogWrapper, ImageObsWrapper
from craftax.craftax_env import make_craftax_env_from_name
from omegaconf import OmegaConf
from agents import get_agent


def main():
    cfg = OmegaConf.load("src/configs/run.yaml")
    cli_overrides = OmegaConf.from_cli()

    cfg = OmegaConf.merge(cfg, cli_overrides)
    agent_inner = OmegaConf.load(cfg.agent_config_path)
    agent_cfg = OmegaConf.create({"agent": agent_inner})
    cfg = OmegaConf.merge(cfg, agent_cfg, cli_overrides)
    print(f"run config:\n{OmegaConf.to_yaml(cfg, resolve=True)}")

    agent = get_agent(cfg)
    env = LogWrapper(ImageObsWrapper(make_craftax_env_from_name(**cfg.env)))
    key = jax.random.key(cfg.seed)
    agent.fit(key, env)


if __name__ == "__main__":
    main()
