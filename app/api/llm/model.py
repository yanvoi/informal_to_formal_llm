from unsloth import FastLanguageModel
from utils import FORMALIZE_PROMPT_PL


class LLMService:
    def __init__(self, model_name: str):
        max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
        dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
        load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.

        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name = model_name,
            max_seq_length = max_seq_length,
            dtype = dtype,
            load_in_4bit = load_in_4bit,
            # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
        )

    def formalize(self, text: str, temperature=0.3) -> str:
        FastLanguageModel.for_inference(self.model)

        inputs = self.tokenizer([FORMALIZE_PROMPT_PL.format(text)], return_tensors = "pt").to("cuda")

        outputs = self.model.generate(**inputs, use_cache=True, temperature=temperature)
        decoded_outputs = self.tokenizer.batch_decode(outputs)
        decoded_outputs = decoded_outputs[-1].split(
            "### Response:\n"
        )[-1].replace("<|end_of_text|>", "")

        return decoded_outputs
