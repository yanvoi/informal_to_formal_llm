import os


FORMALIZE_PROMPT_PL = """
### Instruction:
Przetłumacz poniższy nieformalny tekst w języku polskim na język formalny.

### Input:
{}

### Response:
"""

REPO_ID = os.getenv("REPO_ID", "directtt/Llama-PLLuM-8B-instruct-informal2formal-SFT")
