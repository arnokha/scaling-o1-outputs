## Generate and store responses to prompts
import asyncio
import pandas as pd
from pathlib import Path
from globals import (
    BASE_PATH,
    load_or_create_csv,
    request_o1_chat_completion,
    summarize_task_prompt,
    music_viz_gen_task_prompt,
    log
)

N_RESPONSES_PER_TASK = 12
assert(N_RESPONSES_PER_TASK > 1)
MODEL = "o1-preview"
MODEL_PROVIDER = "openai"

responses_summarize_df_filepath = BASE_PATH / "outputs/1_map_outputs/responses_summarize.csv"
responses_music_viz_df_filepath = BASE_PATH / "outputs/1_map_outputs/responses_music_viz.csv"
responses_columns = ["task_name", "model_provider", "model", "response"]

task_name_summarize = "summarize_intro"
task_name_music_viz = "music_viz_gen"

responses_summarize_df = load_or_create_csv(responses_summarize_df_filepath, responses_columns)
responses_music_viz_df = load_or_create_csv(responses_music_viz_df_filepath, responses_columns)

tasks_assets = [
    # {
    #     "name": task_name_summarize,
    #     "prompt": summarize_task_prompt,
    #     "df": responses_summarize_df,
    #     "filepath": responses_summarize_df_filepath
    # },
    {
        "name": task_name_music_viz,
        "prompt": music_viz_gen_task_prompt,
        "df": responses_music_viz_df,
        "filepath": responses_music_viz_df_filepath
    }
]

semaphore = asyncio.Semaphore(10)  # Limit to 10 concurrent requests
async def fetch_response(task_prompt: str) -> str:
    async with semaphore:
        try:
            response = await request_o1_chat_completion([("user", task_prompt)])
            return response
        except Exception as e:
            log.error(f"Error generating response: {e}")
            return None

## Generate responses to prompts
async def main():
    for task in tasks_assets:
        task_name: str = task["name"]
        task_prompt: str = task["prompt"]
        df: pd.DataFrame = task["df"]
        filepath: Path = task["filepath"]
        
        rows = df.to_dict(orient='records')
        coroutines = []  # To hold coroutines for i > 0

        log.info(f"Processing task: {task_name}")
        print()

        # Warm-up the cache by running the first request synchronously
        warmup_response = await fetch_response(task_prompt)
        if warmup_response is not None:
            new_row = {
                "task_name": task_name,
                "model_provider": MODEL_PROVIDER,
                "model": MODEL,
                "response": warmup_response
            }
            rows.append(new_row)

        # Prepare coroutines for the remaining requests
        coroutines = [fetch_response(task_prompt) for _ in range(1, N_RESPONSES_PER_TASK)]
        # Execute all coroutines concurrently
        responses = await asyncio.gather(*coroutines, return_exceptions=True)
        
        # Handle responses
        for response in responses:
            if isinstance(response, Exception):
                log.error(f"Async task for '{task_name}' raised an exception: {response}")
                continue  # Skip adding this response
            if response is not None:
                new_row = {
                    "task_name": task_name,
                    "model_provider": MODEL_PROVIDER,
                    "model": MODEL,
                    "response": response
                }
                rows.append(new_row)
        
        # Update the DataFrame and save to CSV
        df = pd.DataFrame(rows)
        df.to_csv(filepath, index=False)
        log.info(f"Saved responses for task '{task_name}' to {filepath}")


if __name__ == "__main__":
    asyncio.run(main())