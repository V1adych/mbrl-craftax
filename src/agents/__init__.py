from omegaconf import DictConfig
from .dreamerv3 import DreamerV3

agent_map = {
    "dreamerv3": DreamerV3,
}

def get_agent(config: DictConfig):
    name = config.agent
    assert name in agent_map.keys()
    assert name in config.keys()
    return agent_map[name](config[name])