from typing import Dict
from typing import Optional
import warnings

import optuna

with optuna._imports.try_import() as _imports:
    from keras.callbacks import Callback

import numpy as np


class CustomKerasPruningCallback(Callback):
    """Keras callback to prune unpromising trials.

    See `the example <https://github.com/optuna/optuna-examples/blob/main/
    keras/keras_integration.py>`__
    if you want to add a pruning callback which observes validation accuracy.

    Args:
        trial:
            A :class:`~optuna.trial.Trial` corresponding to the current evaluation of the
            objective function.
        monitor:
            An evaluation metric for pruning, e.g., ``val_loss`` and
            ``val_accuracy``. Please refer to `keras.Callback reference
            <https://keras.io/callbacks/#callback>`_ for further details.
        interval:
            Check if trial should be pruned every n-th epoch. By default ``interval=1`` and
            pruning is performed after every epoch. Increase ``interval`` to run several
            epochs faster before applying pruning.
    """

    log_dict = {}

    def __init__(
        self, trial: optuna.trial.Trial, monitor: str, interval: int = 1
    ) -> None:
        super().__init__()

        _imports.check()

        self._trial = trial
        self._monitor = monitor
        self._interval = interval
        self.log_dict = {}
        self.log_dict["episode_reward"] = []
        self.log_dict["nb_episode_steps"] = []
        self.log_dict["nb_steps"] = []

    def _calculate_custom_metric(self, l: list) -> float:
        max = np.max(l)
        min = np.min(l)
        mean = np.mean(l)

        return float(mean - (max - min) + max)

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, float]] = None) -> None:

        self.log_dict["episode_reward"].append(float(logs["episode_reward"]))
        self.log_dict["nb_episode_steps"].append(int(logs["nb_episode_steps"]))
        self.log_dict["nb_steps"].append(int(logs["nb_steps"]))

        if (epoch + 1) % self._interval != 0:
            return

        logs = logs or {}
        # implement custom metric

        episode_reward_custom = float(
            self._calculate_custom_metric(self.log_dict["episode_reward"])
        )
        nb_steps_custom = float(
            self._calculate_custom_metric(self.log_dict["nb_steps"])
        )

        current_score = float(episode_reward_custom / nb_steps_custom)
        if self.log_dict[self._monitor] is None:
            message = (
                "The metric '{}' is not in the evaluation logs for pruning. "
                "Please make sure you set the correct metric name.".format(
                    self._monitor
                )
            )
            warnings.warn(message)
            return

        self._trial.report(float(current_score), step=epoch)

        self.log_dict["episode_reward"] = []
        self.log_dict["nb_episode_steps"] = []
        self.log_dict["nb_steps"] = []

        if self._trial.should_prune():
            message = "Trial was pruned at epoch {}.".format(epoch)
            raise optuna.TrialPruned(message)
