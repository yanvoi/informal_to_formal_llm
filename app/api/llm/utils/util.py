import os

FORMALIZE_PROMPT_PL = """
Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
Przetłumacz poniższy nieformalny tekst w języku polskim na język formalny.

### Input:
{}

### Response:
"""

REPO_ID = os.getenv("REPO_ID", "directtt/Llama-PLLuM-8B-instruct-informal2formal-SFT")
