from abc import ABC, abstractmethod

from datasets import Dataset
from transformers import AutoTokenizer


class DataPreprocessor(ABC):
    """Abstract Base Class for preprocessing data."""

    @abstractmethod
    def preprocess(self, dataset: Dataset, tokenizer: AutoTokenizer) -> Dataset:
        """Main method to preprocess the data.
        Must be implemented by concrete subclasses.

        Args:
            dataset (Dataset): The dataset to preprocess.
            tokenizer (AutoTokenizer): The tokenizer to use for preprocessing.

        Returns:
            Dataset: The preprocessed dataset.
        """
        pass
