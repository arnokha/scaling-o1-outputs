## Generate and store responses to prompts

from globals import (
    BASE_PATH,
    request_o1_chat_completion,
    summarize_task_prompt,
    music_viz_gen_task_prompt
)

import pandas as pd

N_RESPONSES_PER_TASK = 16
MODEL = "o1-preview"
MODEL_PROVIDER = "openai"

responses_summarize_df_filepath = BASE_PATH / "outputs/1_map_outputs/responses_summarize.csv"
responses_music_viz_df_filepath = BASE_PATH / "outputs/1_map_outputs/responses_music_viz.csv"
responses_columns = ["task_name", "model_provider", "model", "response"]

task_name_summarize = "summarize_intro"
task_name_music_viz = "music_viz_gen"

## TODO create CSVs if they don't exist, otherwise, load them in
# e.g.
# output_file_exists = os.path.isfile(output_file)
# if output_file_exists:
#     df = pd.read_csv(output_file)
# else:
#     df = pd.DataFrame(columns=output_columns)
responses_summarize_df = pd.DataFrame()
responses_music_viz_df = pd.DataFrame()

## TODO log.info the name of the csv and the num rows

for task_name, task_prompt, df in zip(
        [task_name_summarize, task_name_music_viz]
        [summarize_task_prompt, music_viz_gen_task_prompt],
        [responses_summarize_df, responses_music_viz_df]
    ):
        for i in range(N_RESPONSES_PER_TASK):
            response = "test adding results to csv" # await request_o1_chat_completion(("user", task_prompt))
            ## TODO add test result to CSV
            
