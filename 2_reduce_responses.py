##
# Using different "reduce" methods (methods for reducing a set of candidate responses to an ideal response),
# generate optimized versions of the responses.
#
# Reduce methods:
#  - Best of n responses 
#       - LLM chooses best response as final output
#  - Combine n responses
#       - LLM combines responses into an "ideal" final output
#  - (to be implemented) Combine top k of n responses
#       - LLM selects top k responses
#       - LLM combines top k responses into an "ideal" final output
#  - (to be implemented) Pairwise merging responses
#       - Pair responses in their given order
#       - LLM combines responses into an "ideal" intermediate output
#       - Repeat binary fusion of responses until one final "ideal" output remains
#
#  Prompt variations:
#  - 'no_prompt_no_critique': omit original prompt, omit critique instructions
#  - 'no_critique': include original prompt, omit critique instructions
#  - 'with_critique': include original prompt, include critique instructions

import asyncio
import pandas as pd
from pathlib import Path
import os
from openai import AsyncOpenAI
from globals import (
    request_o1_chat_completion,
    experiment_tasks,
    BASE_PATH,
    MODEL,
    MODEL_PROVIDER,
    log,
    replace_placeholders,
    read_file_to_str,
    load_or_create_csv,
    extract_xml,
)
from typing import List, Dict

N_RESPONSES = [2, 4, 8, 16, 32]

# Load and organize prompt templates
reduce_prompts = {
    'best-of-n': {
        'no_prompt_no_critique': read_file_to_str(BASE_PATH / "inputs/prompts/reduce_methods/no_critique_no_orig_prompt/best-of-n.txt"),
        'no_critique': read_file_to_str(BASE_PATH / "inputs/prompts/reduce_methods/no_critique/best-of-n.txt"),
        'with_critique': read_file_to_str(BASE_PATH / "inputs/prompts/reduce_methods/with_critique/best-of-n.txt")
    },
    'combine-n': {
        'no_prompt_no_critique': read_file_to_str(BASE_PATH / "inputs/prompts/reduce_methods/no_critique_no_orig_prompt/combine-n.txt"),
        'no_critique': read_file_to_str(BASE_PATH / "inputs/prompts/reduce_methods/no_critique/combine-n.txt"),
        'with_critique': read_file_to_str(BASE_PATH / "inputs/prompts/reduce_methods/with_critique/combine-n.txt")
    }
}

# Define the tags to extract from the LLM's response for each method
response_extract_tags = {
    'best-of-n': 'best-response',
    'combine-n': 'ideal-response'
}

async def send_reduce_prompt(
    semaphore: asyncio.Semaphore, 
    client: AsyncOpenAI, 
    prompt: str, 
    context: dict
) -> dict:
    async with semaphore:
        try:
            response = await request_o1_chat_completion(client, [("user", prompt)])
            return {'response': response, 'async_call_context': context}
        except Exception as e:
            log.error(f"Error generating reduced response: {e}")
            return {'response': None, 'async_call_context': context}


async def process_reduce_method(
    method_name: str,
    output_csv_fp: Path,
    output_csv_columns: List[str],
    prompt_templates: Dict[str, str],
    response_extract_tag: str,
    semaphore: asyncio.Semaphore,
    client: AsyncOpenAI
):
    # Load or create the output CSV
    output_df = load_or_create_csv(output_csv_fp, output_csv_columns)
    output_rows = output_df.to_dict(orient='records')

    # Prepare a list to hold all coroutines
    coroutines = []

    # Process each task for each value of n
    for task in experiment_tasks:
        task_name: str = task["name"]
        task_prompt: str = task["prompt"]
        responses_df: pd.DataFrame = task["df"]

        for n in N_RESPONSES:
            # Retrieve the top n responses for the task
            responses = responses_df['response'].head(n).tolist()
            # Format the responses with the delimiter
            responses_str = "\n---\n".join(responses)

            # Create mappings for expanding prompt placeholders with data
            responses_replacement_mapping = {
                "{{RESPONSES_FMT}}": responses_str
            }
            responses_replacement_mapping_full = {
                "{{RESPONSES_FMT}}": responses_str,
                "{{PROMPT}}": task_prompt
            }

            for prompt_detail, prompt_template in prompt_templates.items():
                # Create prompt based on prompt_detail variant
                if prompt_detail == 'no_prompt_no_critique':
                    prompt = replace_placeholders(
                        prompt_template, responses_replacement_mapping
                    )
                else:
                    prompt = replace_placeholders(
                        prompt_template, responses_replacement_mapping_full
                    )

                # Prepare context
                async_call_context = {
                    'task_name': task_name,
                    'n': n,
                    'prompt_detail': prompt_detail
                }

                # Schedule the coroutine
                coroutine = send_reduce_prompt(
                    semaphore, client, prompt, context=async_call_context
                )
                coroutines.append(coroutine)

    # Execute all coroutines concurrently
    responses = await asyncio.gather(*coroutines, return_exceptions=False)

    # Process responses
    for res in responses:
        context = res['async_call_context']
        response = res['response']
        task_name = context['task_name']
        n = context['n']
        prompt_detail = context['prompt_detail']

        if response is not None:
            extracted_response = extract_xml(response, response_extract_tag)
            new_row = {
                "task_name": task_name,
                "model_provider": MODEL_PROVIDER,
                "model": MODEL,
                "n": n,
                "prompt_detail": prompt_detail,
                "response": response,
                "extracted_response": extracted_response
            }
            output_rows.append(new_row)

    # Save the DataFrame to CSV
    df = pd.DataFrame(output_rows)
    df.to_csv(output_csv_fp, index=False)
    log.info(f"Saved reduced responses for method '{method_name}' to {output_csv_fp}")

async def main():
    semaphore = asyncio.Semaphore(10)  # Limit concurrent requests
    async with AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY")) as client:
        # Process 'best-of-n' method
        await process_reduce_method(
            method_name='best-of-n',
            output_csv_fp=BASE_PATH / "outputs/2_reduce_outputs/final/best-of-n.csv",
            output_csv_columns=["task_name", "prompt_detail", "model_provider", "model", "n", "response", "extracted_response"],
            prompt_templates=reduce_prompts['best-of-n'],
            response_extract_tag=response_extract_tags['best-of-n'],
            semaphore=semaphore,
            client=client
        )

        # Process 'combine-n' method
        await process_reduce_method(
            method_name='combine-n',
            output_csv_fp=BASE_PATH / "outputs/2_reduce_outputs/final/combine-n.csv",
            output_csv_columns=["task_name", "prompt_detail", "model_provider", "model", "n", "response", "extracted_response"],
            prompt_templates=reduce_prompts['combine-n'],
            response_extract_tag=response_extract_tags['combine-n'],
            semaphore=semaphore,
            client=client
        )


if __name__ == "__main__":
    asyncio.run(main())