import dataclasses
import json
from typing import List, Union

import numpy as np
import torch.optim
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
)
from dac4automlcomp.policy import DACPolicy, DeterministicPolicy


class Serializable:
    """
    Subclass providing a generic way to serialize a dataclass DACPolicy object as a json
    """

    def save(self, path):
        file_path = path.resolve() / f"{self.__class__.__name__}.json"
        with file_path.open(mode="w") as f:
            json.dump(dataclasses.asdict(self), f)

    @classmethod
    def load(cls, path):
        file_path = path.resolve() / f"{cls.__name__}.json"
        with file_path.open(mode="r") as f:
            return cls(**json.load(f))


class Configurable:
    """
    Subclass providing a generic way to specify a DACPolicy's configuration space
    """

    @staticmethod
    def config_space() -> ConfigurationSpace:
        """Return a configuration space object"""
        raise NotImplementedError

    @classmethod
    def from_config(cls, cfg):
        """Return an instance of the class corresponding to the given configuration"""
        return cls(**cfg)


@dataclasses.dataclass
class ConstantLRPolicy(Configurable, Serializable, DeterministicPolicy, DACPolicy):
    """
    A scheduler using the same learning rate throughout the training process.
    """

    lr: float

    def act(self, _):
        return self.lr

    def reset(self, instance):
        pass

    @staticmethod
    def config_space():
        cs = ConfigurationSpace()
        cs.add_hyperparameter(
            UniformFloatHyperparameter("lr", lower=0.000001, upper=10, log=True)
        )
        return cs


@dataclasses.dataclass
class CosineAnnealingLRPolicy(
    Configurable, Serializable, DeterministicPolicy, DACPolicy
):
    """
    Starting from an initial learning rate reduces the learning rate to 0 following a re-scaled half cosine curve.
    """

    lr: float

    def act(self, state):
        return 0.5 * (1 + np.cos(state["step"] * np.pi / self.cutoff)) * self.lr

    def reset(self, instance):
        self.cutoff = instance.cutoff

    @staticmethod
    def config_space():
        cs = ConfigurationSpace()
        cs.add_hyperparameter(
            UniformFloatHyperparameter("lr", lower=0.000001, upper=10, log=True)
        )
        return cs


@dataclasses.dataclass
class SimpleReactivePolicy(Configurable, Serializable, DeterministicPolicy, DACPolicy):
    """
    A novel simple scheduler using a constant learning rate for the first 2 epochs. After every (2+i)th epoch,
    if the average mini-batch loss during the last epoch was lower than that in the last two epochs, it increases
    the learning rate. If the loss is higher, it decreases the learning rate.
    """

    lr: float
    a: float
    b: float

    def act(self, state):
        self.loss += state["loss"].sum()
        if not (state["step"] + 1) % self.epoch_size:  # true at the end of every epoch
            if self.prev_loss is not None:
                self.lr_t *= self.a if self.loss < self.prev_loss else 1.0 / self.b
            self.prev_loss = self.loss
            self.loss = 0.0
        return self.lr_t

    def reset(self, instance):
        self.lr_t = self.lr
        self.loss = 0.0
        self.prev_loss = None
        self.epoch_size = len(instance.loaders[0])

    @staticmethod
    def config_space():
        cs = ConfigurationSpace()
        cs.add_hyperparameters(
            [
                UniformFloatHyperparameter("lr", lower=0.000001, upper=10, log=True),
                UniformFloatHyperparameter("1/a", lower=0.1, upper=1.0),
                UniformFloatHyperparameter("1/b", lower=0.1, upper=1.0),
            ]
        )
        return cs

    @classmethod
    def from_config(cls, cfg):
        cfg = dict(**cfg.get_dictionary())
        cfg["a"] = 1.0 / cfg.pop("1/a")
        cfg["b"] = 1.0 / cfg.pop("1/b")
        return cls(**cfg)


@dataclasses.dataclass
class ReduceLROnPlateauPolicy(
    Configurable, Serializable, DeterministicPolicy, DACPolicy
):
    """
    A scheduler emulating the ReduceLROnPlateau pytorch scheduler. It adjusts the learning rate per epoch,
    based on the validation loss (observed at the end of every epoch). In particular, it reduces the learning rate
    with a `factor' if the validation loss has `stagnated' for `patience' epochs.
    """

    lr: float
    mode: str = "min"
    factor: float = 0.1
    patience: int = 10
    threshold: float = 1e-4
    threshold_mode: str = "rel"
    cooldown: int = 0
    min_lr: Union[float, List[float]] = 0
    eps: float = 1e-8

    def act(self, state):
        if state["validation_loss"] is not None:  # true at the end of every epoch
            self.scheduler.step(state["validation_loss"].mean())
        return self.optimizer.param_groups[0]["lr"]

    def reset(self, _):
        self.scheduler, self.optimizer = self.__create_scheduler(
            **dataclasses.asdict(self)
        )

    @staticmethod
    def __create_scheduler(*, lr, **scheduler_params):
        optimizer = torch.optim.SGD([torch.nn.Parameter(torch.tensor(0.0))], lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, **scheduler_params
        )
        return scheduler, optimizer

    @staticmethod
    def config_space():
        cs = ConfigurationSpace()
        cs.add_hyperparameters(
            [
                UniformFloatHyperparameter("lr", lower=0.000001, upper=10, log=True),
                UniformFloatHyperparameter("factor", lower=0.1, upper=1.0),
                UniformIntegerHyperparameter("patience", lower=1, upper=10),
            ]
        )
        return cs


@dataclasses.dataclass
class MotoCosineAnnealingLRPolicy(
    Configurable, Serializable, DeterministicPolicy, DACPolicy
):
    """
    Starting from an initial learning rate reduces the learning rate to 0 following a re-scaled half cosine curve.
    """

    lr: float

    from scipy.interpolate import interp1d

    a = [0, 20, 30, 50, 80, 100]
    b = [0.99, 0.3, 0.26, 0.22, 0.15, 0.1]
    finter = interp1d(a, b, kind='cubic')

    def act(self, state):
        # LB: 2.7000 ({"lr": 0.004172917577117374})
        # return 0.5 * (1 + np.cos(state["step"] * np.pi / self.cutoff)) * self.lr

        def sigmoid(x):
            f_x = 1 / (1 + np.exp(-x))
            return f_x

        def lr_sigmoid(step_ratio):
            x = step_ratio - 0.5
            step_proportion = 1 - sigmoid(x * 10)
            return step_proportion

        def lr_sigmoid3(step_ratio, factor=5):
            x = step_ratio - 0.5
            step_proportion = 1 - sigmoid(x * factor)
            return step_proportion * 0.8

        def lr_inter(step_ratio):
            return self.finter(step_ratio * 100)

        def lr_inter2(step_ratio):
            return 2*self.finter(step_ratio * 100)

        def lr_cos(ratio, MAX_LR=1):
            alpha = ratio * np.pi
            lr = 0.5 * (1 + np.cos(alpha)) * MAX_LR
            return lr

        def lr_sigmoid2(step_ratio, factor=5):
            x = step_ratio - 0.5
            step_proportion = 1 - sigmoid(x * factor)
            return step_proportion

        def lr_ens(step_ratio):
            if step_ratio <= 0.5:
                return lr_cos(step_ratio)
            else:
                return lr_sigmoid2(step_ratio)

        def lr_ens3(step_ratio):
            if step_ratio >= 0.5:
                return lr_cos(step_ratio)
            else:
                return lr_sigmoid2(step_ratio, factor=6)

        ratio = state["step"] / self.cutoff
        proportion = lr_ens(ratio)

        return proportion * self.lr

    def reset(self, instance):
        self.cutoff = instance.cutoff

    @staticmethod
    def config_space():
        cs = ConfigurationSpace()
        cs.add_hyperparameter(
            UniformFloatHyperparameter("lr", lower=0.000001, upper=10, log=True)
        )
        return cs
