import jax
from craftax.craftax_env import make_craftax_env_from_name
from omegaconf import OmegaConf
from agents import get_agent


def main():
    config = OmegaConf.load("src/configs/run.yaml")
    cli_overrides = OmegaConf.from_cli()
    config = OmegaConf.merge(config, cli_overrides)
    print(OmegaConf.to_yaml(config, resolve=True))

    agent = get_agent(config)
    env = make_craftax_env_from_name(**config.env)
    key = jax.random.key(config.seed)
    agent.fit(key, env)


if __name__ == "__main__":
    main()
