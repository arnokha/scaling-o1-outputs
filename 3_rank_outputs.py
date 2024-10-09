## judge outputs from reduce methods, use one original response as a control
#  NOTE: does not record the original model used to generate, nor the ranking model, TODO fix
#  NOTE: unlike 1_ and 2_, this script overwrites the CSV instead of appending to it if it exists
import asyncio
import pandas as pd
from pathlib import Path
import os
import random
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
    extract_xml,
)

#N_VALUES = [2, 4, 8, 16, 32]
N_VALUES = [2, 4, 8, 16]
N_PERMUTAIONS_RANK = 5 # rerank with different orderings in an attempt to unbias preference for earlier entries

BEST_OF_N_REDUCE_OUTPUTS_PATH = BASE_PATH / "outputs/2_reduce_outputs/final/best-of-n.csv"
COMBINE_N_REDUCE_OUTPUTS_PATH = BASE_PATH / "outputs/2_reduce_outputs/final/combine-n.csv"
RANKING_PROMPT_PATH = BASE_PATH / "inputs/prompts/ranking.txt"


def get_control_response(task_df):
    control_response = task_df.iloc[0]['response']
    return control_response


def get_reduced_responses_for_task_n(best_of_n_df, combine_n_df, task_name, n):
    responses = []

    # Get responses from best-of-n method
    for prompt_detail in ['no_prompt_no_critique', 'no_critique', 'with_critique']:
        # Filter DataFrame
        rows = best_of_n_df[
            (best_of_n_df['task_name'] == task_name) &
            (best_of_n_df['n'] == n) &
            (best_of_n_df['prompt_detail'] == prompt_detail)
        ]
        # Use first entry row per combination
        if not rows.empty:
            response = rows.iloc[0]['extracted_response']
            # Store details
            responses.append({
                'method': 'best-of-n',
                'prompt_detail': prompt_detail,
                'n': n,
                'response': response
            })
        else:
            log.warning(f"No response found for best-of-n, task {task_name}, n={n}, prompt_detail={prompt_detail}")
   
    # Get responses from combine-n method
    for prompt_detail in ['no_prompt_no_critique', 'no_critique', 'with_critique']:
        # Filter DataFrame
        rows = combine_n_df[
            (combine_n_df['task_name'] == task_name) &
            (combine_n_df['n'] == n) &
            (combine_n_df['prompt_detail'] == prompt_detail)
        ]
        # Use first entry row per combination
        if not rows.empty:
            response = rows.iloc[0]['extracted_response']
            # Store details
            responses.append({
                'method': 'combine-n',
                'prompt_detail': prompt_detail,
                'n': n,
                'response': response
            })
        else:
            log.warning(f"No response found for combine-n, task {task_name}, n={n}, prompt_detail={prompt_detail}")
    return responses


def format_responses_for_prompt(perm):
    responses_str = ''
    for idx_in_prompt, (original_idx, response_dict) in enumerate(perm):
        # The index needs to be 1-based (since the LLM might be more familiar with that)
        response_index = idx_in_prompt + 1
        response_text = response_dict['response']
        responses_str += f"Response {response_index}:\n{response_text}\n---\n"
    return responses_str.strip('---\n')  # Remove the trailing '---\n'


async def send_ranking_prompt(semaphore, client, prompt, context):
    async with semaphore:
        try:
            response = await request_o1_chat_completion(client, [("user", prompt)])
            return {'response': response, 'context': context}
        except Exception as e:
            log.error(f"Error generating ranking response: {e}")
            return {'response': None, 'context': context}

# Compare same n: vary prompt_detail and reduce method (control is an original response without reduction)
async def process_same_n_comparison(semaphore, client, best_of_n_df, combine_n_df, ranking_prompt_template):
    coroutines = []
    for task in experiment_tasks:
        task_name = task['name']
        task_prompt = task['prompt']
        task_df = task['df']
        control_response = get_control_response(task_df)
        for n in N_VALUES:
            responses = get_reduced_responses_for_task_n(best_of_n_df, combine_n_df, task_name, n)
            # Include the control response
            responses.append({
                'method': 'control',
                'prompt_detail': 'original',
                'n': n,
                'response': control_response
            })
            # Assign indexes (0-based)
            responses_with_indexes = list(enumerate(responses))  # [(0, response_dict), (1, response_dict), ...]
            # Generate permutations
            for perm_id in range(N_PERMUTAIONS_RANK):
                perm = responses_with_indexes.copy()
                random.shuffle(perm)
                # Format the responses
                responses_formatted = format_responses_for_prompt(perm)
                # Prepare replacements
                replacement_mapping = {
                    '{{PROMPT}}': task_prompt,
                    '{{RESPONSES_FMT}}': responses_formatted
                }
                prompt = replace_placeholders(ranking_prompt_template, replacement_mapping)
                # Prepare context
                context = {
                    'task_name': task_name,
                    'n': n,
                    'perm_id': perm_id,
                    'perm': perm  # Store the permuted list of (original_idx, response_dict)
                }
                coroutine = send_ranking_prompt(semaphore, client, prompt, context)
                coroutines.append(coroutine)

    # Execute all coroutines concurrently
    responses = await asyncio.gather(*coroutines, return_exceptions=False)

    # Initialize an empty list to collect ranking results
    ranking_results = []

    for res in responses:
        response_text = res['response']
        context = res['context']
        task_name = context['task_name']
        n = context['n']
        perm_id = context['perm_id']
        perm = context['perm']

        if response_text is not None:
            # Extract the <ranking> tag
            ranking_str = extract_xml(response_text, 'ranking')
            # Convert ranking_str to list of indexes
            ranking_str = ranking_str.strip()
            # Expected format is comma-separated list of indexes, e.g., "3,5,2,1,4"
            ranking_list = [int(s.strip()) for s in ranking_str.split(',') if s.strip().isdigit()]
            # Now, map back to the methods
            # The original indexes are 1-based in the prompt (we displayed to LLM)
            # So the positions in perm are 0-based
            for rank, idx_in_ranking in enumerate(ranking_list):
                # idx_in_ranking is 1-based position in permuted list
                idx_in_perm_list = idx_in_ranking - 1  # Adjust to 0-based index
                if idx_in_perm_list >= len(perm):
                    log.error(f"Index in ranking ({idx_in_ranking}) exceeds number of responses ({len(perm)}).")
                    continue
                original_idx, response_dict = perm[idx_in_perm_list]
                method = response_dict['method']
                prompt_detail = response_dict['prompt_detail']
                # Record the result
                ranking_results.append({
                    'task_name': task_name,
                    'n': n,
                    'perm_id': perm_id,
                    'method': method,
                    'prompt_detail': prompt_detail,
                    'rank': rank + 1,  # Rank is 1-based
                })
        else:
            log.error(f"No response for task {task_name}, n={n}, perm_id={perm_id}")

    # Create DataFrame from ranking_results
    ranking_df = pd.DataFrame(ranking_results)
    # Compute average rank for each task, n, method, prompt_detail
    average_ranks = ranking_df.groupby(['task_name', 'n', 'method', 'prompt_detail'])['rank'].mean().reset_index()

    # Save the ranking results and average ranks to CSV
    ranking_df.to_csv(BASE_PATH / "outputs/3_rank_outputs/ranking_results_same_n.csv", index=False)
    average_ranks.to_csv(BASE_PATH / "outputs/3_rank_outputs/average_ranks_same_n.csv", index=False)
    log.info("Saved ranking results and average ranks for same n comparison.")

# Compare same reduction method: vary n responses (control is an original response without reduction)
async def process_across_n_comparison(semaphore, client, best_of_n_df, combine_n_df, ranking_prompt_template):
    coroutines = []
    for task in experiment_tasks:
        task_name = task['name']
        task_prompt = task['prompt']
        task_df = task['df']
        control_response = get_control_response(task_df)
        for method in ['best-of-n', 'combine-n']:
            for prompt_detail in ['no_prompt_no_critique', 'no_critique', 'with_critique']:
                responses = []

                df = best_of_n_df if method == 'best-of-n' else combine_n_df
                for n in N_VALUES:
                    # Filter DataFrame
                    rows = df[
                        (df['task_name'] == task_name) &
                        (df['n'] == n) &
                        (df['prompt_detail'] == prompt_detail)
                    ]
                    if not rows.empty:
                        response = rows.iloc[0]['extracted_response']
                        # Store details
                        responses.append({
                            'method': method,
                            'prompt_detail': prompt_detail,
                            'n': n,
                            'response': response
                        })
                    else:
                        log.warning(f"No response found for {method}, task {task_name}, n={n}, prompt_detail={prompt_detail}")

                # Include the control response
                responses.append({
                    'method': 'control',
                    'prompt_detail': 'original',
                    'n': 'control',
                    'response': control_response
                })

                # Assign indexes
                responses_with_indexes = list(enumerate(responses))
                # Generate permutations
                for perm_id in  range(N_PERMUTAIONS_RANK):
                    perm = responses_with_indexes.copy()
                    random.shuffle(perm)
                    # Format the responses
                    responses_formatted = format_responses_for_prompt(perm)
                    # Prepare replacements
                    replacement_mapping = {
                        '{{PROMPT}}': task_prompt,
                        '{{RESPONSES_FMT}}': responses_formatted
                    }
                    prompt = replace_placeholders(ranking_prompt_template, replacement_mapping)
                    # Prepare context
                    context = {
                        'task_name': task_name,
                        'method': method,
                        'prompt_detail': prompt_detail,
                        'perm_id': perm_id,
                        'perm': perm  # Store the permuted list of (original_idx, response_dict)
                    }
                    coroutine = send_ranking_prompt(semaphore, client, prompt, context)
                    coroutines.append(coroutine)

    # Execute all coroutines concurrently
    responses = await asyncio.gather(*coroutines, return_exceptions=False)

    # Initialize an empty list to collect ranking results
    ranking_results = []

    for res in responses:
        response_text = res['response']
        context = res['context']
        task_name = context['task_name']
        method = context['method']
        prompt_detail = context['prompt_detail']
        perm_id = context['perm_id']
        perm = context['perm']

        if response_text is not None:
            # Extract the <ranking> tag
            ranking_str = extract_xml(response_text, 'ranking')
            # Convert ranking_str to list of indexes
            ranking_str = ranking_str.strip()
            # Expected format is comma-separated list of indexes, e.g., "3,5,2,1,4"
            ranking_list = [int(s.strip()) for s in ranking_str.split(',') if s.strip().isdigit()]
            # Now, map back to the methods
            # The original indexes are 1-based in the prompt (we displayed to LLM)
            # So the positions in perm are 0-based
            for rank, idx_in_ranking in enumerate(ranking_list):
                # idx_in_ranking is 1-based position in permuted list
                idx_in_perm_list = idx_in_ranking - 1  # Adjust to 0-based index
                if idx_in_perm_list >= len(perm):
                    log.error(f"Index in ranking ({idx_in_ranking}) exceeds number of responses ({len(perm)}).")
                    continue
                original_idx, response_dict = perm[idx_in_perm_list]
                n = response_dict['n']
                # Record the result
                ranking_results.append({
                    'task_name': task_name,
                    'method': method,
                    'prompt_detail': prompt_detail,
                    'n': n,
                    'perm_id': perm_id,
                    'rank': rank + 1,  # Rank is 1-based
                })
        else:
            log.error(f"No response for task {task_name}, method={method}, prompt_detail={prompt_detail}, perm_id={perm_id}")

    # Create DataFrame from ranking_results
    ranking_df = pd.DataFrame(ranking_results)
    # Compute average rank for each task, method, prompt_detail, n
    average_ranks = ranking_df.groupby(['task_name', 'method', 'prompt_detail', 'n'])['rank'].mean().reset_index()

    # Save the ranking results and average ranks to CSV
    ranking_df.to_csv(BASE_PATH / "outputs/3_rank_outputs/ranking_results_across_n.csv", index=False)
    average_ranks.to_csv(BASE_PATH / "outputs/3_rank_outputs/average_ranks_across_n.csv", index=False)
    log.info("Saved ranking results and average ranks for across n comparison.")


async def main():
    # Read necessary data
    best_of_n_df = pd.read_csv(BEST_OF_N_REDUCE_OUTPUTS_PATH)
    combine_n_df = pd.read_csv(COMBINE_N_REDUCE_OUTPUTS_PATH)
    ranking_prompt_template = read_file_to_str(RANKING_PROMPT_PATH)

    semaphore = asyncio.Semaphore(10)  # Limit concurrent requests
    async with AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY")) as client:
        # Process same n comparison
        await process_same_n_comparison(semaphore, client, best_of_n_df, combine_n_df, ranking_prompt_template)
        # Process across n comparison
        await process_across_n_comparison(semaphore, client, best_of_n_df, combine_n_df, ranking_prompt_template)


if __name__ == "__main__":
    asyncio.run(main())