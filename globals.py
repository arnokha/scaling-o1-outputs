import os
import pandas as pd
import logging
from openai import AsyncOpenAI
from typing import List, Tuple, Optional, Dict
from pathlib import Path
from custom_logging import get_logger_with_level

log = get_logger_with_level( logging.DEBUG )

OPEN_AI_API_KEY = os.environ.get("OPENAI_API_KEY")
async_openai_client = AsyncOpenAI(api_key=OPEN_AI_API_KEY)

BASE_PATH = Path(os.path.dirname(os.path.realpath(__file__)))

# roles are assistant and user
async def request_o1_chat_completion(
    msgs: List[Tuple[str,str]], 
    model: str = "o1-preview",
)-> Optional[str]:
    """Request exam guide enhancement chat completion"""
    messages = [{"role": role, "content": msg_content} for role, msg_content in msgs]
    try:
        completion = await async_openai_client.chat.completions.create(
            messages=messages,
            model=model,
        )
        content = completion.choices[0].message.content   

    except Exception as e:
        model_provider = "OpenAI"
        log.error(f"Error while requesting {model_provider} chat completion: {e}")
        raise e
    return str(content)

def read_file_to_text(filepath: str) -> str:
    try:
        with open(filepath, 'r') as f:
            text = f.read()
    except Exception as e:
        raise IOError(f"Error reading file {filepath}: {e}")   
    return text

def replace_placeholders(text: str, replacements: Dict[str, str]) -> str:
    """Replace placeholders in instruction and user input templates"""
    for key, value in replacements.items():
        text = text.replace(key, value)
    return text

def load_or_create_csv(filepath, columns) -> pd.DataFrame:
    """
    Load a CSV file if it exists, otherwise create a new DataFrame with specified columns.

    Args:
        filepath (Path): Path to the CSV file.
        columns (list): List of column names for the DataFrame.

    Returns:
        pd.DataFrame: Loaded or newly created DataFrame.
    """
    if filepath.exists():
        df = pd.read_csv(filepath)
        log.info(f"Loaded existing CSV: {filepath.name} with {len(df)} rows.")
    else:
        df = pd.DataFrame(columns=columns)
        log.info(f"Created new DataFrame for CSV: {filepath.name}.")
    return df
# def parse_reasoning_and_answer(response_text: str) -> Tuple[str, str]:
#     # Split the response text based on the delimiter (three or more dashes)
#     parts = re.split(r'-{3,}', response_text)
    
#     if len(parts) != 2:
#         raise ValueError("Invalid response format. Single delimiter not found.")
    
#     # Strip leading/trailing whitespace from reasoning and answer
#     reasoning = parts[0].strip()
#     answer = parts[1].strip()
    
#     return (reasoning, answer)

# def extract_winner(response: str) -> int:
#     if not response:
#         raise ValueError("The input response is empty.")
    
#     lines = response.strip().split('\n')
#     for line in reversed(lines):
#         if 'winner' in line.lower():
#             numbers = re.findall(r'\d+', line)
#             if numbers:
#                 winner_number = int(numbers[0])
#                 if winner_number in {1, 2}:
#                     return winner_number
#                 else:
#                     raise ValueError("Invalid winner number: must be 1 or 2.")
#             else:
#                 raise ValueError("No number found in the winner declaration line.")
    
#     raise ValueError("No winner declaration found in the response.")


## Global vars
summarize_task_prompt = read_file_to_text(BASE_PATH / "inputs/prompts/tasks/summarize.txt")
music_viz_gen_task_prompt = read_file_to_text(BASE_PATH / "inputs/prompts/tasks/music_viz_gen.txt")

summarize_task_reference = read_file_to_text(BASE_PATH / "inputs/references/grayling_introduction.txt")
music_viz_gen_task_reference = read_file_to_text(BASE_PATH / "inputs/references/music_viz_gen.py")

summarize_task_prompt = replace_placeholders(summarize_task_prompt, {"{{REFERENCE}}": summarize_task_reference})
music_viz_gen_task_prompt = replace_placeholders(music_viz_gen_task_prompt, {"{{REFERENCE}}": music_viz_gen_task_reference})