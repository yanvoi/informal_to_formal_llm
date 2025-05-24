from dagshub.data_engine import datasources
from datasets import Dataset
import pandas as pd
from transformers import PreTrainedTokenizer

from informal_to_formal.data_preprocessor.data_preprocessor import DataPreprocessor
from informal_to_formal.utils.consts import ALPACA_PROMPT_TEMPLATE


class TrainingDataDataPreprocessor(DataPreprocessor):
    """Handles loading and preprocessing of datasets for training.

    Attributes:
        dataset_uri (str): The URI of the DagsHub dataset to load.
        prompt_template (str): The template for formatting prompts.
    """

    def __init__(
        self,
        dataset_uri: str,
        prompt_template: str = ALPACA_PROMPT_TEMPLATE,
    ):
        self.dataset_uri = dataset_uri
        self.prompt_template = prompt_template

    def _formatting_prompts_func(self, examples: Dataset, eos_token: str) -> dict[str, list[str]]:
        inputs = examples["zdanie_nieformalne"]
        targets = examples["zdanie_formalne"]
        formatted = []

        for inp, tgt in zip(inputs, targets):
            text = self.prompt_template.format(inp, tgt)
            formatted.append(text + eos_token)

        return {"text": formatted}

    def get_raw_dataframe(self, split: str) -> pd.DataFrame:
        """Get the raw dataframe for a specific split.

        Args:
            split (str): The split to retrieve ("train", "val", or "test").

        Returns:
            pd.DataFrame: The raw dataframe for the specified split.
        """
        ds = datasources.get("informal2formal/mlflow", self.dataset_uri)
        df = pd.read_csv(ds.head().dataframe["dagshub_download_url"].values[0])

        return df[df["split"] == split][["zdanie_nieformalne", "zdanie_formalne"]]

    def load_data(self) -> tuple[Dataset, Dataset, Dataset]:
        """Load the dataset from DagsHub and split it into train, validation, and test sets.

        Returns:
            tuple[Dataset, Dataset, Dataset]: A tuple containing the train, validation, and test datasets.
        """
        splits = {
            "train": self.get_raw_dataframe("train"),
            "val": self.get_raw_dataframe("val"),
            "test": self.get_raw_dataframe("test"),
        }

        return (
            Dataset.from_pandas(splits["train"]),
            Dataset.from_pandas(splits["val"]),
            Dataset.from_pandas(splits["test"]),
        )

    def preprocess(
        self,
        dataset: Dataset,
        tokenizer: PreTrainedTokenizer,
    ) -> Dataset:
        """Preprocess the dataset by formatting the prompts.

        Args:
            dataset (Dataset): The dataset to preprocess.
            tokenizer (PreTrainedTokenizer): The tokenizer to use for tokenization.

        Returns:
            Dataset: The preprocessed dataset.
        """
        eos_token = tokenizer.eos_token
        return dataset.map(
            lambda examples: self._formatting_prompts_func(examples, eos_token),
            batched=True,
        )
