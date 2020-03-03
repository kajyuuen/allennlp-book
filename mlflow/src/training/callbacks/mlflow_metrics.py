import logging
import os
import time

import torch
import mlflow
from allennlp.common.checks import ConfigurationError
from allennlp.models import Model
from allennlp.training.callbacks.callback import Callback, handle_event
from allennlp.training.callback_trainer import CallbackTrainer
from allennlp.training.callbacks.events import Events
from allennlp.training.metric_tracker import MetricTracker


logger = logging.getLogger(__name__)


@Callback.register("mlflow_metrics")
class MlflowMetrics(Callback):
    def __init__(self, should_log_learning_rate: bool = False) -> None:
        self._should_log_learning_rate = should_log_learning_rate

    @staticmethod
    def log_metric(key, value, step=None):
        if mlflow.active_run() is None:
            logger.warning("A new mlflow active run will be created.")
        mlflow.log_metric(key, value, step)

    @handle_event(Events.BATCH_END, priority=100)
    def end_of_batch(self, trainer: CallbackTrainer):
        step = trainer.batch_num_total
        self.log_metric("batch/training_loss", trainer.train_metrics["loss"], step)
        for key, value in trainer.train_metrics.items():
            self.log_metric("batch/training_" + key, value, step)

        if self._should_log_learning_rate:
            names = {param: name for name, param in trainer.model.named_parameters()}
            for group in trainer.optimizer.param_groups:
                if "lr" not in group:
                    continue
                rate = group["lr"]
                for param in group["params"]:
                    effective_rate = rate * float(param.requires_grad)
                    self.log_metric(
                        "batch/learning_rate." + names[param],
                        effective_rate, step
                    )

    @handle_event(Events.EPOCH_END, priority=110)
    def end_of_epoch(self, trainer: CallbackTrainer):
        epoch = trainer.epoch_number + 1
        training_elapsed_time = time.time() - trainer.training_start_time
        self.log_metric("training_duration_seconds", training_elapsed_time, epoch)

        for key, value in trainer.train_metrics.items():
            self.log_metric("epoch/training_" + key, value, epoch)
        for key, value in trainer.val_metrics.items():
            self.log_metric("epoch/validation_" + key, value, epoch)
