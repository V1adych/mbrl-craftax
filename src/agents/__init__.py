from omegaconf import DictConfig
from .dreamerv3 import DreamerV3
from .twm import IRIS

agent_map = {
    "dreamerv3": DreamerV3,
    "twm_iris": IRIS,
}

def get_agent(config: DictConfig):
    name = config.agent_name
    assert name in agent_map.keys()
    return agent_map[name](config.agent)