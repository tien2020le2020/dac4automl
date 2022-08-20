from pathlib import Path
import dataclasses
import json
from typing import List, Union

import numpy as np
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
    CategoricalHyperparameter,
)
from dac4automlcomp.policy import DACPolicy, DeterministicPolicy
from baselines.schedulers import Configurable, Serializable


@dataclasses.dataclass
class MotoSchedulePolicy(Configurable, Serializable, DeterministicPolicy, DACPolicy):
    algorithm: str
    learning_rates: List[float]
    gammas: List[float]
    gae_lambdas: List[float]
    vf_coefs: List[float]
    ent_coefs: List[float]
    clip_ranges: List[float]
    buffer_sizes: List[int]
    learning_starts: List[int]
    batch_sizes: List[int]
    taus: List[float]
    train_freqs: List[int]
    gradient_steps: List[int]

    # env: str
    # t: int
    # discrete: bool

    def act(self, state):
        env = self.env

        if env == "CARLPendulumEnv":  # Zoo
            action = {
                "algorithm": "PPO",

                # Zoo
                # "learning_rate": 3e-4, # Current BEST
                "learning_rate": 3e-4 * 1.5,  # New

                "n_steps": 2048,

                "gamma": 0.99,
                "gae_lambda": 0.95,
                "ent_coef": 0.0,
                "batch_size": 64,
            }
        elif env == "CARLAcrobotEnv":  # Zoo
            action = {
                "algorithm": "PPO",

                "gamma": 0.99,
                "gae_lambda": 0.94,
                "ent_coef": 0.0,
                "n_epochs": 4,

                # "learning_rate": 1e-4 * 0.5,  # Current BEST
                "learning_rate": 1e-4 * 0.35,  # New
            }
        elif env == "CARLCartPoleEnv":
            action = {
                "algorithm": "PPO",

                # "n_epochs": 10,
                # Tuned
                # "learning_rate": 0.001,  # Current BEST
                "learning_rate": 0.001 * 0.65,  # New

                "gamma": 0.98,
                "gae_lambda": 0.8,
                "ent_coef": 0.0,
                "n_steps": 32,
                "n_epochs": 20,
                "batch_size": 256,
            }
        elif env == "CARLLunarLanderEnv":  # Zoo - OK
            action = {
                "algorithm": "PPO",

                "n_steps": 1024,
                "batch_size": 64,
                "gae_lambda": 0.98,
                "gamma": 0.999,
                "n_epochs": 4,
                "ent_coef": 0.01,

                "learning_rate": 4e-5 * 0.25,  # BEST - fixed
            }
        elif env == "CARLMountainCarContinuousEnv":  # PB2: OK
            if self.t < len(self.learning_rates):
                self.t += 1

            if self.algorithm == "PPO":
                action = {
                    "algorithm": self.algorithm,
                    "learning_rate": self.learning_rates[min(self.t, len(self.learning_rates) - 1)],
                    "gamma": self.gammas[min(self.t, len(self.gammas) - 1)],
                    "gae_lambda": self.gae_lambdas[min(self.t, len(self.gae_lambdas) - 1)],
                    "vf_coef": self.vf_coefs[min(self.t, len(vf_coefs) - 1)],
                    "ent_coef": self.ent_coefs[min(self.t, len(ent_coefs) - 1)],
                    "clip_range": self.clip_ranges[min(self.t, len(self.clip_ranges) - 1)],
                }
                if self.t > 0:
                    del action["clip_range"]
            else:
                action = {
                    "algorithm": self.algorithm,
                    "learning_rate": self.learning_rates[min(self.t, len(self.learning_rates) - 1)],
                    "buffer_size": self.buffer_sizes[min(self.t, len(self.buffer_sizes) - 1)],
                    "learning_starts": self.learning_starts[min(self.t, len(self.learning_starts) - 1)],
                    "batch_size": self.batch_sizes[min(self.t, len(self.batch_sizes) - 1)],
                    "tau": self.taus[min(self.t, len(self.taus) - 1)],
                    "gamma": self.gammas[min(self.t, len(self.gammas) - 1)],
                    "train_freq": self.train_freqs[min(self.t, len(self.train_freqs) - 1)],
                    "gradient_steps": self.gradient_steps[min(self.t, len(self.gradient_steps) - 1)],
                }
                if self.t > 0:
                    del action["train_freq"]

        return action

    def reset(self, instance):
        self.t = 0
        if instance[0] in ['CARLLunarLanderEnv', 'CARLCartPoleEnv', 'CARLAcrobotEnv']:
            self.discrete = True
        else:
            self.discrete = False

        self.env = instance.env_type

    @staticmethod
    def config_space():
        cs = ConfigurationSpace()
        cs.add_hyperparameter(
            UniformFloatHyperparameter(
                "learning_rate", lower=0.000001, upper=10, log=True
            )
        )
        cs.add_hyperparameter(
            UniformFloatHyperparameter("gamma", lower=0.000001, upper=1.0)
        )
        cs.add_hyperparameter(
            UniformFloatHyperparameter("gae_lambda", lower=0.000001, upper=0.99)
        )
        cs.add_hyperparameter(
            UniformFloatHyperparameter("vf_coef", lower=0.000001, upper=1.0)
        )
        cs.add_hyperparameter(
            UniformFloatHyperparameter("ent_coef", lower=0.000001, upper=1.0)
        )
        cs.add_hyperparameter(
            UniformFloatHyperparameter("clip_range", lower=0.0, upper=1.0)
        )
        cs.add_hyperparameter(UniformFloatHyperparameter("tau", lower=0.0, upper=1.0))
        cs.add_hyperparameter(
            UniformIntegerHyperparameter("buffer_size", lower=1000, upper=1e8)
        )
        cs.add_hyperparameter(
            UniformIntegerHyperparameter("learning_starts", lower=1, upper=1e4)
        )
        cs.add_hyperparameter(
            UniformIntegerHyperparameter("batch_size", lower=8, upper=1024)
        )
        cs.add_hyperparameter(
            UniformIntegerHyperparameter("train_freq", lower=1, upper=1e4)
        )
        cs.add_hyperparameter(
            UniformIntegerHyperparameter("gradient_steps", lower=-1, upper=1e3)
        )
        # CategoricalHyperparameter("algorithm", choices=["PPO", "SAC", "DDPG"])
        cs.add_hyperparameter(
            CategoricalHyperparameter("algorithm", choices=["PPO", "SAC", "DDPG"])
        )
        return cs


def load_solution(
    policy_cls=MotoSchedulePolicy, path=Path(".")
) -> DACPolicy:
    """
    Load Solution.
    Serves as an entry point for the competition evaluation.
    By default (the submission) it loads a saved SMAC optimized configuration for the ConstantLRPolicy.
    Args:
        policy_cls: The DACPolicy class object to load
        path: Path pointing to the location the DACPolicy is stored
    Returns
    -------
    DACPolicy
    """
    path = Path(path, 'logs', 'pb2_seed0', 'saved_configs')
    return policy_cls.load(path)
