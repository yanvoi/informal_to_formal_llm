from abc import ABC, abstractmethod


class Trainer(ABC):
    """Abstract Base Class for training models."""

    @abstractmethod
    def train(self, experiment_name: str, run_name: str):
        """Main method to train the model.

        Args:
            experiment_name (str): Name of the mlflow experiment.
            run_name (str): Name of the mlflow run.
        """
        pass
