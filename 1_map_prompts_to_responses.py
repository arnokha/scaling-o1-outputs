## Generate and store responses to task prompts
import asyncio
import pandas as pd
from pathlib import Path
import os
from openai import AsyncOpenAI
from globals import (
    request_o1_chat_completion,
    experiment_tasks,
    MODEL,
    MODEL_PROVIDER,
    log
)

N_RESPONSES_PER_TASK = 16
assert(N_RESPONSES_PER_TASK > 1)

async def fetch_response(semaphore: asyncio.Semaphore, client: AsyncOpenAI, task_prompt: str) -> str:
    async with semaphore:
        try:
            response = await request_o1_chat_completion(client, [("user", task_prompt)])
            return response
        except Exception as e:
            log.error(f"Error generating response: {e}")
            return None

## Generate responses to prompts
async def main():
    semaphore = asyncio.Semaphore(10)  # Limit to 10 concurrent requests
    async with AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY")) as client:
        for task in experiment_tasks:
            task_name: str = task["name"]
            task_prompt: str = task["prompt"]
            df: pd.DataFrame = task["df"]
            responses_filepath: Path = task["responses_filepath"]
            
            rows = df.to_dict(orient='records') # keep records from existing df
            coroutines = []  # To hold coroutines for i > 0

            log.info(f"Processing task: {task_name}")
            print()

            # Populate the cache by running the first request synchronously
            warmup_response = await fetch_response(semaphore, client, task_prompt)
            if warmup_response is not None:
                new_row = {
                    "task_name": task_name,
                    "model_provider": MODEL_PROVIDER,
                    "model": MODEL,
                    "response": warmup_response
                }
                rows.append(new_row)

            # Prepare coroutines for the remaining requests
            coroutines = [fetch_response(semaphore, client, task_prompt) for _ in range(1, N_RESPONSES_PER_TASK)]
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
            df.to_csv(responses_filepath, index=False)
            log.info(f"Saved responses for task '{task_name}' to {responses_filepath}")


if __name__ == "__main__":
    asyncio.run(main())