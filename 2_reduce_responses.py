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
    reduce_prompt__best_of_n__no_prompt_no_critique,
    reduce_prompt__best_of_n__no_critique,
    reduce_prompt__best_of_n__with_critique,
    load_or_create_csv,
    extract_xml,
    reduce_prompt__combine_n__no_prompt_no_critique,
    reduce_prompt__combine_n__no_critique,
    reduce_prompt__combine_n__no_prompt_no_critique
)

N_VALUES = [2, 4, 8]
## Using the different reduce methods, and given varied outputs to a prompt,
## generate an optimized version of the prompt

## - Best of n responses (w and w/o critique)
## - Combine n responses (w and w/o critique)
## - Combine to k of n responses (w and w/o critique)
## - Pairwise merging (w and w/o critique)

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


async def main():
    semaphore = asyncio.Semaphore(10)  # Limit to 10 concurrent requests
    async with AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY")) as client:
        
        ##-=-=-=-=-=-=-=-=-=
        ## Best of n
        ##-=-=-=-=-=-=-=-=-=
        best_of_n_df_fp = BASE_PATH / f"outputs/2_reduce_outputs/final/best-of-n.csv"
        best_of_n_df_columns = ["task_name", "prompt_detail", "model_provider", "model", "n", "response", "best_answer"]
        best_of_n_df = load_or_create_csv(best_of_n_df_fp, best_of_n_df_columns)
        best_of_n_rows = best_of_n_df.to_dict(orient='records')
        reduce_coroutines = []
        
        # Run best-of-n on all tasks and n values
        for task in experiment_tasks:
            task_name: str = task["name"]
            task_prompt: str = task["prompt"]
            responses_df: pd.DataFrame = task["df"]
            
            for n in N_VALUES:
                async_call_context = {
                    'task_name': task_name,
                    'n': n,
                }
                # Retrieve the top n responses for task
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
                
                # Create different versions of expanded prompt
                prompt_no_prompt_no_critique = replace_placeholders(
                    reduce_prompt__best_of_n__no_prompt_no_critique, responses_replacement_mapping
                )
                prompt_no_critique = replace_placeholders(
                    reduce_prompt__best_of_n__no_critique, responses_replacement_mapping_full
                )
                prompt_with_critique = replace_placeholders(
                    reduce_prompt__best_of_n__with_critique, responses_replacement_mapping_full
                )
                
                # Create reduce coroutines for different versions of expanded prompt
                no_prompt_no_critique_context = async_call_context.copy()
                no_prompt_no_critique_context["prompt_detail"] = "no_prompt_no_critique"
                reduce_coroutines.append(
                    send_reduce_prompt(
                        semaphore, client, prompt_no_prompt_no_critique,
                        context = no_prompt_no_critique_context
                    )
                )
                
                no_critique_context = async_call_context.copy()
                no_critique_context["prompt_detail"] = "no_critique"
                reduce_coroutines.append(
                    send_reduce_prompt(
                        semaphore, client, prompt_no_critique,
                        context = no_critique_context
                    )
                )
                
                with_critique_context = async_call_context.copy()
                with_critique_context["prompt_detail"] = "with_critique"
                reduce_coroutines.append(
                    send_reduce_prompt(
                        semaphore, client, prompt_with_critique,
                        context = with_critique_context
                    )
                )

            # Execute all coroutines concurrently
            reduce_responses = await asyncio.gather(*reduce_coroutines, return_exceptions=False)

            # Process all responses - extract data to columns
            for res in reduce_responses:
                context = res['async_call_context']
                response = res['response']
                
                task_name = context['task_name']
                n = context['n']
                prompt_detail = context['prompt_detail']

                if response is not None:
                    new_row = {
                        "task_name": task_name,
                        "model_provider": MODEL_PROVIDER,
                        "model": MODEL,
                        "n": n,
                        "prompt_detail": prompt_detail,
                        "response": response,
                        "response_best_answer": extract_xml(response, "best-response")
                    }
                    best_of_n_rows.append(new_row)

        # Save all DataFrames to their respective CSV files
        df = pd.DataFrame(best_of_n_rows)
        df.to_csv(best_of_n_df_fp, index=False)
        log.info(f"Saved reduced responses to {best_of_n_df_fp}")
        
        # TODO more reduce methods
            
            

            
            
if __name__ == "__main__":
    asyncio.run(main())
    
    