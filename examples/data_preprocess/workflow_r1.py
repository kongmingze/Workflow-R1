import os
import logging
import tempfile
import argparse
import pandas as pd
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import EntryNotFoundError

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

system_content = """You are a helpful assistant who is an expert in reasoning and problem solving."""
prompt_template = '''Answer the given question. Every time you get new information from an operator, you must conduct reasoning inside <think> and </think> before taking any further action. This means you should have multiple thinking sessions - one after each time you receive information from operators, not just one initial thinking session.

After reasoning, if you find you can utilize an operator to help you increase the likelihood of obtaining a correct answer, you can call an operator according to the operator description below. And when you call the operator, you MUST follow the corresponding format.

Available Operators: {operators}

Before each operator call, you must explicitly reason inside <think> and </think> about why you're choosing this operator, how you'll use it, and what you expect to learn.
When you call an operator, the response will be returned between <info> and </info>. After receiving this response, you must again think inside <think> and </think> about what the information means, how it helps answer the original question, and what your next step should be.

CRITICAL INSTRUCTION: Do not settle for simple, single-step operator calls. You are REQUIRED to aggressively explore diverse, complex, and unconventional combinations of operators. Your goal is not just to answer, but to exhaustively explore how different operators can synergize. You must push the boundaries of what these tools can achieve together. Passive, lazy, or simplistic strategies are unacceptable; you must demonstrate deep reasoning through rich, multi-step operator orchestration to uncover their unique capabilities and "magical" combined effects.
IMPORTANT CONSTRAINT: You have a MAXIMUM of 20 operator calls. Plan your strategy wisely - you must gather sufficient information and reach a confident answer within this limit.

Always follow this cycle: think, then optionally use an operator if needed, then when you receive information, think again, then decide your next step. Repeat this cycle until you have enough information to answer confidently.
After you have gathered enough information and conducted deep reasoning through multiple thinking cycles, if you are truly certain you can answer the query correctly, directly provide your final answer in only a few words or a short phrase inside <answer> and </answer> to exactly answer the query. If you're not certain, continue gathering information and reasoning until you reach certainty.

Query: {query}'''

operator_description = """1. AnswerGenerate: Call an advanced LLM to think and answer questions. Useful for domain-specific questions, knowledge QA, reasoning tasks, etc. Usage: <tool>AnswerGenerate: YOUR_QUESTION</tool>
2. Custom: Call an LLM to execute any task you define. As long as you provide a clear and detailed instruction along with the task content, the LLM can accomplish virtually anything you need. Usage: <tool>Custom: <task>YOUR_TASK_CONTENT</task><instruction>YOUR_INSTRUCTION</instruction></tool>
3. PromptOptimizer: Call an LLM to refine and optimize a query's clarification by adding details and creating a step-by-step plan. Useful when the original query is ambiguous or needs more context for better understanding. This operator can help you polish and optimize your query and you will get a better one to use. Usage: <tool>PromptOptimizer: YOUR_QUERY</tool>
4. Review: Call an LLM to critically review a solution's correctness and provide feedback. Useful when you want to verify whether some answer is correct before finalizing. You can utilize to verify some answer and get feedback, then decide how to do next. Usage: <tool>Review: <question>ORIGINAL_QUESTION</question><solution>SOLUTION_TO_REVIEW</solution></tool>
5. Revise: Call an LLM to revise a solution based on feedback. Useful when you have received critical feedback indicating there are something wrong in the solution. You can use the operator to have better solution revised according to the feedback information you provide. Usage: <tool>Revise: <question>ORIGINAL_QUESTION</question><solution>SOLUTION_TO_REVISE</solution><feedback>FEEDBACK_RECEIVED</feedback></tool>
6. MultiAnswerGenerate: Call an advanced LLM to generate multiple independent answers (3 times) for the same question. Useful when you want to explore diverse reasoning paths, gather multiple perspectives, or prepare candidates for ensemble selection. Each answer is generated separately to ensure diversity. The results are concatenated and labeled as [Answer 1], [Answer 2], [Answer 3]. Usage: <tool>MultiAnswerGenerate: YOUR_QUESTION</tool>
7. ScEnsemble: Call an LLM to select the best answer from multiple candidate solutions through self-consistency ensemble selection. Useful when you have generated multiple possible answers for the same question and want to pick the most reliable one by comparing and voting among them. This operator leverages collective wisdom from diverse solutions to improve final answer quality and reduce randomness in single-shot generation. You must provide at least two candidate solutions wrapped in separate <solution></solution> tags for meaningful ensemble comparison. Usage: <tool>ScEnsemble: <query>ORIGINAL_QUESTION</query><solution>CANDIDATE_ANSWER_1</solution>...<solution>CANDIDATE_ANSWER_N</solution></tool>"""


def process_single_row(row, current_split_name, row_index):
    question = row.get("question", "")
    
    user_content = prompt_template.format(operators=operator_description, query=question)
    prompt = [{"role": "system", "content": system_content}, {"role": "user", "content": user_content}]

    reward_model_data = row.get("reward_model")
    if isinstance(reward_model_data, dict) and "ground_truth" in reward_model_data:
        ground_truth = reward_model_data.get("ground_truth")
    else:
        ground_truth = row.get("golden_answers", [])

    extra_info = {
        'split': current_split_name,
        'answer': ground_truth,
        "question": question,
    }

    return pd.Series({
        "data_source": "workflow_r1",
        "data_original_source": row.get("data_source", ""),
        'agent_name': 'workflow_r1_agent_loop',
        "prompt": prompt,
        "ability": row.get("ability"),
        "reward_model": reward_model_data,
        "extra_info": extra_info,
        "metadata": row.get("metadata"),
    })


def main():
    local_save_dir = os.path.expanduser(args.local_dir)
    os.makedirs(local_save_dir, exist_ok=True)
    processed_files = []

    with tempfile.TemporaryDirectory() as tmp_download_dir:
        for split in ["train", "test"]:
            parquet_filename = f"{split}.parquet"
            logger.info(f"Processing {split} split...")

            try:
                logger.info(f"Downloading {parquet_filename} from {args.hf_repo_id}")
                local_parquet_filepath = hf_hub_download(
                    repo_id=args.hf_repo_id,
                    filename=parquet_filename,
                    repo_type="dataset",
                    local_dir=tmp_download_dir,
                    local_dir_use_symlinks=False,
                )

                df_raw = pd.read_parquet(local_parquet_filepath)
                logger.info(f"Loaded {len(df_raw)} rows from {parquet_filename}")

                if split == "train":
                    max_samples = 10000
                    if len(df_raw) > max_samples:
                        logger.info(f"Sampling {split} split from {len(df_raw)} to {max_samples} samples")
                        df_raw = df_raw.sample(n=max_samples, random_state=42)
                    else:
                        logger.info(f"{split} split has {len(df_raw)} samples, no sampling needed")
                else:
                    logger.info(f"Using all {len(df_raw)} samples for {split} split")

                df_processed = df_raw.apply(
                    lambda row: process_single_row(row, current_split_name=split, row_index=row.name), 
                    axis=1
                )

                output_file_path = os.path.join(local_save_dir, f"{split}.parquet")
                df_processed.to_parquet(output_file_path, index=False)
                logger.info(f"Saved {len(df_processed)} processed rows to {output_file_path}")
                processed_files.append(output_file_path)

            except EntryNotFoundError:
                logger.warning(f"{parquet_filename} not found in repository {args.hf_repo_id}")
            except Exception as e:
                logger.error(f"Error processing {split} split: {e}")

    if not processed_files:
        logger.warning("No data was processed or saved")
        return

    logger.info(f"Successfully processed {len(processed_files)} files to {local_save_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and process dataset from HuggingFace.")
    parser.add_argument("--hf_repo_id", default="PeterJinGo/nq_hotpotqa_train", help="HuggingFace dataset repository ID.")
    parser.add_argument("--local_dir", default="./GSsPO", help="Local directory to save processed files.")
    args = parser.parse_args()