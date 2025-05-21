from transformers import PreTrainedModel, PreTrainedTokenizer
from unsloth import FastLanguageModel

from informal_to_formal.utils.consts import ALPACA_PROMPT_TEMPLATE


def generate_language_model_output(
    input: str,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    alpaca_prompt: str = ALPACA_PROMPT_TEMPLATE,
    temperature=0.3,
) -> str:
    """Generate output from a language model using a prompt template.

    Args:
        input (str): The input text to be processed by the model.
        model (PreTrainedModel): The language model to be used for generation.
        tokenizer (PreTrainedTokenizer): The tokenizer for the language model.
        alpaca_prompt (str): The prompt template to be used for generation.
        temperature (float): The temperature to use.
            Defaults to 0.3.

    Returns:
        str: The generated output from the model.
    """
    FastLanguageModel.for_inference(model)
    inputs = tokenizer(
        [
            alpaca_prompt.format(
                input,  # input
                "",  # output - leave this blank for generation!
            )
        ],
        return_tensors="pt",
    ).to("cuda")

    outputs = model.generate(**inputs, use_cache=True, temperature=temperature)
    decoded_outputs = tokenizer.batch_decode(outputs)
    decoded_outputs = (
        decoded_outputs[-1].split("### Response:\n")[-1].replace("<|end_of_text|>", "")
    )

    return decoded_outputs
