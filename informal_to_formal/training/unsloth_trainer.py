import logging
from typing import Any

import dagshub
import mlflow
import pandas as pd
from datasets import Dataset
from tqdm import tqdm
from trainer import Trainer
from transformers import AutoTokenizer, TrainingArguments
from trl import SFTTrainer
from unsloth import FastLanguageModel, is_bfloat16_supported

from informal_to_formal.data_preprocessor import TrainingDataDataPreprocessor
from informal_to_formal.evaluation import Evaluator
from informal_to_formal.utils.consts import ALPACA_PROMPT_TEMPLATE, TEST_DATA_LIMIT
from informal_to_formal.utils.inference import generate_language_model_output


class UnslothTrainer(Trainer):
    """Trainer class for training a language model using the unsloth.ai framework.
    This class handles the loading of the model, dataset, and training configuration.
    It also manages the training process and logging of results to MLFlow.

    Attributes:
        model_name (str): Name of the base model to be used.
        dataset_uri (str): URI for the training dataset.
        prompt_template (str): Template for the prompts.
        max_seq_length (int): Maximum sequence length for the model.
        per_device_train_batch_size (int): Batch size per device during training.
        gradient_accumulation_steps (int): Number of gradient accumulation steps.
        warmup_steps (int): Number of warmup steps for learning rate scheduling.
        num_train_epochs (int): Number of training epochs.
        eval_steps (int): Evaluation steps during training.
        learning_rate (float): Learning rate for the optimizer.
        weight_decay (float): Weight decay for the optimizer.
        lr_scheduler_type (str): Type of learning rate scheduler to use.
        seed (int): Random seed for reproducibility.
        extra_args (dict): Additional arguments for training.
    """

    def __init__(
        self,
        model_name: str,
        dataset_uri: str,
        prompt_template: str = ALPACA_PROMPT_TEMPLATE,
        max_seq_length: int = 2048,
        per_device_train_batch_size: int = 2,
        gradient_accumulation_steps: int = 4,
        warmup_steps: int = 15,
        num_train_epochs: int = 1,
        eval_steps: int = 100,
        learning_rate: float = 2e-4,
        weight_decay: float = 0.01,
        lr_scheduler_type: str = "linear",
        seed: int = 3407,
        **kwargs: Any,
    ):
        # Hyperparameters
        self.model_name = model_name
        self.dataset_uri = dataset_uri
        self.prompt_template = prompt_template
        self.max_seq_length = max_seq_length
        self.per_device_train_batch_size = per_device_train_batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.warmup_steps = warmup_steps
        self.num_train_epochs = num_train_epochs
        self.eval_steps = eval_steps
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.lr_scheduler_type = lr_scheduler_type
        self.seed = seed
        self.extra_args = kwargs

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(self.__class__.__name__)

        dagshub.init(repo_owner="informal2formal", repo_name="mlflow", mlflow=True)
        self.data_preprocessor = TrainingDataDataPreprocessor(
            dataset_uri=dataset_uri,
            prompt_template=prompt_template,
        )

        self.model: FastLanguageModel = None
        self.tokenizer: AutoTokenizer = None

    def _load_base_model(self):
        """Load the base model and tokenizer."""
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            self.model_name,
            max_seq_length=self.max_seq_length,
            dtype=None,
            load_in_4bit=True,
        )

    def _load_and_preprocess_data(
        self
    ) -> tuple[Dataset, Dataset, Dataset]:
        """Load and preprocess the training, validation, and test datasets."""
        train_ds, val_ds, test_ds = self.data_preprocessor.load_data()

        train_ds = self.data_preprocessor.preprocess(train_ds, self.tokenizer)
        val_ds = self.data_preprocessor.preprocess(val_ds, self.tokenizer)
        test_ds = self.data_preprocessor.preprocess(test_ds, self.tokenizer)

        return train_ds, val_ds, test_ds

    def _apply_peft(self) -> FastLanguageModel:
        """Apply PEFT (Parameter-Efficient Fine-Tuning) to the base model."""
        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r=16,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            lora_alpha=16,
            lora_dropout=0,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=self.seed,
            use_rslora=False,
            loftq_config=None,
        )
        return self.model

    def _get_training_config(self, run_name: str) -> TrainingArguments:
        """Get the training configuration for the SFTTrainer."""
        return TrainingArguments(
            per_device_train_batch_size=self.per_device_train_batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            warmup_steps=self.warmup_steps,
            num_train_epochs=self.num_train_epochs,
            eval_strategy="steps",
            eval_steps=self.eval_steps,
            learning_rate=self.learning_rate,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=self.weight_decay,
            lr_scheduler_type=self.lr_scheduler_type,
            seed=self.seed,
            output_dir=run_name,
            report_to="mlflow",
            **self.extra_args,
        )

    def _evaluate_on_test_dataset(self) -> tuple[pd.DataFrame, dict]:
        """Evaluate the model on the test dataset."""
        test_df = self.data_preprocessor.get_raw_dataframe("test")
        test_list_pred = test_df["zdanie_nieformalne"].tolist()[:TEST_DATA_LIMIT]
        test_list_target = test_df["zdanie_formalne"].tolist()[:TEST_DATA_LIMIT]
        predictions = [
            generate_language_model_output(prompt, self.model, self.tokenizer)
            for prompt in tqdm(test_list_pred)
        ]

        evaluator = Evaluator(
            pd.DataFrame(
                {
                    "pred": predictions,
                    "target": test_list_target,
                }
            )
        )
        evaluate_df_metrics, avg_metrics = evaluator.evaluate()

        return evaluate_df_metrics, avg_metrics

    def train(self, experiment_name: str, run_name: str):
        """Main method to train the model using the SFTTrainer.

        Args:
            experiment_name (str): Name of the mlflow experiment.
            run_name (str): Name of the mlflow run.

        Returns:
            TODO
        """
        # Load base model and tokenizer
        self._load_base_model()
        self.logger.info(f"Loaded base model: {self.model_name}")

        # Load and preprocess datasets for training
        train_ds, val_ds, test_ds = self._load_and_preprocess_data()
        self.logger.info("Loaded and preprocessed train/val/test datasets.")

        # Initialize SFTTrainer
        trainer = SFTTrainer(
            model=self._apply_peft(),
            tokenizer=self.tokenizer,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            dataset_text_field="text",
            max_seq_length=self.max_seq_length,
            dataset_num_proc=2,
            args=self._get_training_config(run_name),
        )
        self.logger.info("Trainer with PEFT config initialized.")

        mlflow.set_experiment(experiment_name)
        with mlflow.start_run(run_name=run_name):
            stats = trainer.train()

            # Log base parameters
            mlflow.log_param("base_model_name", self.model_name)
            mlflow.log_param("prompt_template", self.prompt_template)
            mlflow.log_param("dataset_uri", self.dataset_uri)

            # Evaluate on test dataset
            test_df_metrics, avg_metrics = self._evaluate_on_test_dataset()
            mlflow.log_metrics(avg_metrics)

        return stats  # TODO: return stats, test_df_metrics, avg_metrics?
