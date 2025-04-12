from evaluate import load
import pandas as pd
from tqdm import tqdm

from informal_to_formal.utils.consts import (
    EVALUATION_DATA_PRED_COLUMN,
    EVALUATION_DATA_TARGET_COLUMN,
)


class Evaluator:
    """
    A class to evaluate the performance of a model using metrics like ROUGE and BERTScore.

    Attributes:
        evaluation_data (pd.DataFrame): DataFrame containing the evaluation data
            with columns for predictions and targets.
    """

    def __init__(self, evaluation_data: pd.DataFrame):
        self.evaluation_data = evaluation_data

        self._validate_evaluation_data()

    def _validate_evaluation_data(self):
        """Validates the evaluation data to ensure that the necessary columns are present
        and that they contain valid text data.

        Raises:
            ValueError: If any of the required columns are missing or contain invalid data.
        """

        # Check if required columns are present
        required_columns = [EVALUATION_DATA_PRED_COLUMN, EVALUATION_DATA_TARGET_COLUMN]
        for column in required_columns:
            if column not in self.evaluation_data.columns:
                raise ValueError(f"Missing required column: '{column}'")

        # Check if the columns contain valid text data
        for column in required_columns:
            if not self.evaluation_data[column].apply(lambda x: isinstance(x, str)).all():
                raise ValueError(f"Column '{column}' must contain only strings.")

            if self.evaluation_data[column].apply(lambda x: len(x.strip()) == 0).any():
                raise ValueError(f"Column '{column}' contains empty or whitespace-only strings.")

    def _rouge_score(self, use_stemmer: bool = False) -> (pd.DataFrame, dict):
        """Calculate the ROUGE score for a given DataFrame.

        Args:
            use_stemmer (bool): Whether to use stemming for the ROUGE score calculation. Defaults to False.

        Returns:
            pd.DataFrame: DataFrame with ROUGE scores for each row.
            dict: Dictionary with overall ROUGE score.
        """
        rouge = load("rouge")

        # compute the ROUGE score for each row individually
        tqdm.pandas(desc="Computing ROUGE scores")

        def compute_row_rouge(row):
            result = rouge.compute(
                predictions=[row[EVALUATION_DATA_PRED_COLUMN]],
                references=[[row[EVALUATION_DATA_TARGET_COLUMN]]],
                use_stemmer=use_stemmer,
            )

            return result["rouge1"], result["rouge2"], result["rougeL"]

        # apply the function to each row and store the ROUGE scores in new columns
        self.evaluation_data[["rouge1", "rouge2", "rougeL"]] = self.evaluation_data.progress_apply(
            lambda row: pd.Series(compute_row_rouge(row)), axis=1
        )

        # calculate the overall average ROUGE scores
        overall_results = {
            "rouge1": float(self.evaluation_data["rouge1"].mean()),
            "rouge2": float(self.evaluation_data["rouge2"].mean()),
            "rougeL": float(self.evaluation_data["rougeL"].mean()),
        }

        return self.evaluation_data, overall_results

    def _bert_score(self, target_lang: str = "pl") -> (pd.DataFrame, dict):
        """Calculate the BERTScore for a given DataFrame.

        Args:
            target_lang (str): The language of the text. Defaults to "pl".

        Returns:
            pd.DataFrame: DataFrame with BERT scores for each row.
            dict: Dictionary with overall BERT score.
        """
        bert_score = load("bertscore")

        # compute BERT score for each row individually
        tqdm.pandas(desc="Computing BERT scores")

        def compute_row_bert_score(row):
            # For each row, wrap the reference in an inner list
            result = bert_score.compute(
                predictions=[row[EVALUATION_DATA_PRED_COLUMN]],
                references=[[row[EVALUATION_DATA_TARGET_COLUMN]]],
                lang=target_lang,
            )

            return result["precision"][0], result["recall"][0], result["f1"][0]

        # apply the function to each row and store the BERT scores in new columns
        self.evaluation_data[["bert_precision", "bert_recall", "bert_f1"]] = (
            self.evaluation_data.progress_apply(
                lambda row: pd.Series(compute_row_bert_score(row)), axis=1
            )
        )

        # calculate the overall average BERT scores
        overall_results = {
            "bert_precision": float(self.evaluation_data["bert_precision"].mean()),
            "bert_recall": float(self.evaluation_data["bert_recall"].mean()),
            "bert_f1": float(self.evaluation_data["bert_f1"].mean()),
        }

        return self.evaluation_data, overall_results

    def evaluate(self, use_stemmer: bool = False, target_lang: str = "pl") -> (pd.DataFrame, dict):
        """
        Evaluate the model using ROUGE and BERTScore metrics.

        Args:
            use_stemmer:
                Whether to use stemming for the ROUGE score calculation. Defaults to False.
            target_lang:
                The language of the text for BERTScore. Defaults to "pl".

        Returns:
            pd.DataFrame: DataFrame with evaluation scores for each row.
            dict: Dictionary with overall evaluation scores.
        """

        # Calculate ROUGE scores
        self.evaluation_data, avg_rouge = self._rouge_score(use_stemmer=use_stemmer)

        # Calculate BERT scores
        self.evaluation_data, avg_bert = self._bert_score(target_lang=target_lang)

        return self.evaluation_data, {**avg_rouge, **avg_bert}
