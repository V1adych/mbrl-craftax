from omegaconf import DictConfig
import jax
from gymnax.environments.environment import Environment
from craftax.craftax_classic.envs.craftax_state import EnvState
from .models import Dynamics, Encoder, Decoder, Posterior, Prior, Actor, Critic

class DreamerV3:
    def __init__(self, config: DictConfig):
        self.config = config
    
    def _init_(self, env: Environment, env_state: EnvState):
        actspace = env.action_space(env_state).n
        self.dynamics = Dynamics(self.config.dynamics, actspace)
        self.encoder = Encoder(self.config.encoder)
        self.decoder = Decoder(self.config.decoder)
        self.posterior = Posterior(self.config.posterior)
        self.prior = Prior(self.config.prior)
        self.actor = Actor(self.config.actor, actspace)
        self.critic = Critic(self.config.critic)

    def fit(self, key: jax.Array, env: Environment):
        pass
