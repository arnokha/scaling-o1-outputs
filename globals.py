# utils and globals
import os
import pandas as pd
import logging
from typing import List, Tuple, Optional, Dict
from pathlib import Path
from custom_logging import get_logger_with_level
from openai import AsyncOpenAI

##-=-=-=-=-=-=-=-=-=-=
## Helper functions
##-=-=-=-=-=-=-=-=-=-=
async def request_o1_chat_completion(
    client: AsyncOpenAI,
    msgs: List[Tuple[str,str]], # roles are assistant and user
    model: str = "o1-preview",
)-> Optional[str]:
    """Request openai o1 chat completion"""
    messages = [{"role": role, "content": msg_content} for role, msg_content in msgs]
    try:
        completion = await client.chat.completions.create(
            messages=messages,
            model=model,
        )
        content = completion.choices[0].message.content   

    except Exception as e:
        model_provider = "OpenAI"
        log.error(f"Error while requesting {model_provider} chat completion: {e}")
        raise e
    return str(content)

def read_file_to_str(filepath: str) -> str:
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
    """Load a CSV file if it exists, otherwise create a new DataFrame with specified columns."""
    if filepath.exists():
        df = pd.read_csv(filepath)
        log.info(f"Loaded existing CSV: {filepath.name} with {len(df)} rows.")
    else:
        df = pd.DataFrame(columns=columns)
        log.info(f"Created new DataFrame for CSV: {filepath.name}.")
    return df

def extract_xml(response: str, tag: str):
    opening_tag = f"<{tag}>"
    closing_tag = f"</{tag}>"
    if not (opening_tag in response and closing_tag in response):
        log.warning("Unable to extract information for given tag (LLM likely did not follow response formatting instructions)")
        return response
    else:
        start_index = response.find(opening_tag) + len(opening_tag)
        end_index = response.find(closing_tag, start_index)
        content = response[start_index:end_index]
        return content.strip()

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

##-=-=-=-=-=-=-=-=-=-=
## Global vars
##-=-=-=-=-=-=-=-=-=-=
log = get_logger_with_level( logging.WARNING )

BASE_PATH = Path(os.path.dirname(os.path.realpath(__file__)))

summarize_task_prompt = read_file_to_str(BASE_PATH / "inputs/prompts/tasks/summarize.txt")
music_viz_gen_task_prompt = read_file_to_str(BASE_PATH / "inputs/prompts/tasks/music_viz_gen.txt")

summarize_task_reference = read_file_to_str(BASE_PATH / "inputs/references/grayling_introduction.txt")
music_viz_gen_task_reference = read_file_to_str(BASE_PATH / "inputs/references/music_viz_gen.py")

summarize_task_prompt = replace_placeholders(summarize_task_prompt, {"{{REFERENCE}}": summarize_task_reference})
music_viz_gen_task_prompt = replace_placeholders(music_viz_gen_task_prompt, {"{{REFERENCE}}": music_viz_gen_task_reference})

responses_summarize_df_filepath = BASE_PATH / "outputs/1_map_outputs/responses_summarize.csv"
responses_music_viz_df_filepath = BASE_PATH / "outputs/1_map_outputs/responses_music_viz.csv"
responses_columns = ["task_name", "model_provider", "model", "response"]

MODEL = "o1-preview"
MODEL_PROVIDER = "openai"

task_name_summarize = "summarize_intro"
task_name_music_viz = "music_viz_gen"

responses_summarize_df = load_or_create_csv(responses_summarize_df_filepath, responses_columns)
responses_music_viz_df = load_or_create_csv(responses_music_viz_df_filepath, responses_columns)

experiment_tasks = [
    {
        "name": task_name_summarize,
        "prompt": summarize_task_prompt,
        "df": responses_summarize_df,
        "responses_filepath": responses_summarize_df_filepath
    },
    {
        "name": task_name_music_viz,
        "prompt": music_viz_gen_task_prompt,
        "df": responses_music_viz_df,
        "responses_filepath": responses_music_viz_df_filepath
    }
]

reduce_prompt__best_of_n__no_prompt_no_critique = read_file_to_str(BASE_PATH / "inputs/prompts/reduce_methods/no_critique_no_orig_prompt/best-of-n.txt") # only need to insert responses
reduce_prompt__best_of_n__no_critique = read_file_to_str(BASE_PATH / "inputs/prompts/reduce_methods/no_critique/best-of-n.txt") # insert responses and orig prompt
reduce_prompt__best_of_n__with_critique = read_file_to_str(BASE_PATH / "inputs/prompts/reduce_methods/with_critique/best-of-n.txt") # insert responses and orig prompt

reduce_prompt__combine_n__no_prompt_no_critique = read_file_to_str(BASE_PATH / "inputs/prompts/reduce_methods/no_critique_no_orig_prompt/combine-n.txt") # only need to insert responses
reduce_prompt__combine_n__no_critique = read_file_to_str(BASE_PATH / "inputs/prompts/reduce_methods/no_critique/combine-n.txt") # insert responses and orig prompt
reduce_prompt__combine_n__with_critique = read_file_to_str(BASE_PATH / "inputs/prompts/reduce_methods/with_critique/combine-n.txt") # insert responses and orig prompt
