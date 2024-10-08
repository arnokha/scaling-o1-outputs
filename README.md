# LLM Response Optimization Experiment

This repository contains scripts to conduct an experiment on optimizing responses generated by Large Language Models (LLMs). The experiment involves generating multiple responses to given prompts, reducing them via various methods to produce optimized responses, and ranking these outputs to evaluate the effectiveness of the reduction methods.

The scripts are intended to be run in order:

1. `1_map_prompts_to_responses.py`
2. `2_reduce_responses.py`
3. `3_rank_outputs.py`

## Overview
This experiment aims to test methods for improving the quality of LLM-generated responses by:

1. **Generating multiple responses** to specific tasks.
2. **Applying reduction methods** to produce optimized responses:
   - **Best-of-N**: Selecting the best response out of N generated responses.
   - **Combine-N**: Combining N responses into an ideal response.
3. **Ranking the reduced outputs**, including a control (original response without reduction), to evaluate the performance of the reduction methods.

**Tasks in this experiment:**

- **Summarization Task**: Summarize an introduction text.
- **Code Generation Task**: Generate code for music visualization.

## Usage

Run the scripts in order.

### 1. Generate Responses to Prompts

```bash
python 1_map_prompts_to_responses.py
```

This script:

- Generates multiple responses for each task prompt using the OpenAI API.
- Saves the responses to CSV files in `outputs/1_map_outputs/`.

### 2. Reduce Responses Using Specified Methods

```bash
python 2_reduce_responses.py
```

This script applies reduction methods to the generated responses:

- **Best-of-N**: Selects the best response out of N.
- **Combine-N**: Combines N responses into an ideal response.

It uses different prompt variations:

- `no_prompt_no_critique`: Omits both the original prompt and critique instructions.
- `no_critique`: Includes the original prompt but omits critique instructions.
- `with_critique`: Includes both the original prompt and critique instructions.

Processes various values of **N** `[2, 4, 8, 16, 32]`.

Reduced responses are saved to CSV files in `outputs/2_reduce_outputs/final/`.

### 3. Rank the Outputs

```bash
python 3_rank_outputs.py
```

This script:

- Ranks the reduced outputs, including a control response (original response without reduction).
- Uses an LLM to obtain rankings. Runs multiple permutations of ordering to reduce any bias from ordering of responses.
- Saves the ranking results to CSV files in `outputs/3_rank_outputs/`.

## Outputs

After running the scripts, you will find:

### Generated Responses (`outputs/1_map_outputs/`)

- `responses_summarize.csv`: Responses for the summarization task.
- `responses_music_viz.csv`: Responses for the music visualization task.

### Reduced Responses (`outputs/2_reduce_outputs/final/`)

- `best-of-n.csv`: Reduced responses using the Best-of-N method.
- `combine-n.csv`: Reduced responses using the Combine-N method.

### Ranking Results (`outputs/3_rank_outputs/`)

- `ranking_results_same_n.csv`: Ranking results comparing methods for the same N. (vary and compare prompt_detail and reduce method)
- `average_ranks_same_n.csv`: Average ranks for the same N comparison.
- `ranking_results_across_n.csv`: Ranking results comparing N values for the same method. (vary and compare n responses)
- `average_ranks_across_n.csv`: Average ranks for the across N comparison.

## Results (2024/10/09)
- "Combine-n", with critique, and a small n responses (n=2,4), seems to work best.
- Should probably be run with more tasks for a more trustworthy result