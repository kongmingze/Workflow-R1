# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Workflow Reward Score Function

用于计算 Workflow 任务的 reward
"""

import re
import json
from typing import Dict, Any, Callable, List, Tuple
from math import isclose
import asyncio
import math

import random
import re
import string
from collections import Counter

import multiprocessing

from . import prime_math
from . import Call_LLM
from verl.workers.reward_manager import similarity_reward

def extract_solution(solution_str):
    answer_pattern = r"<answer>(.*?)</answer>"
    match = re.finditer(answer_pattern, solution_str, re.DOTALL)
    matches = list(match)

    if len(matches) < 1:
        return None

    return matches[-1].group(1).strip()

def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def subem_check(prediction, golden_answers):
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    score = 0
    for golden_answer in golden_answers:
        golden_answer = normalize_answer(golden_answer)
        if golden_answer in normalized_prediction:
            score = 1
            break
    return score


def compute_score_subem(solution_str, ground_truth, method="strict", format_score=0.0, score=1.0):
    """The scoring function for substring exact match (EM).

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """
    answer = extract_solution(solution_str=solution_str)
    score = subem_check(answer, ground_truth["target"])
    return score

def em_check(prediction, golden_answers):
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    score = 0
    for golden_answer in golden_answers:
        golden_answer = normalize_answer(golden_answer)
        if golden_answer == normalized_prediction:
            score = 1
            break
    return score

def f1_score(prediction: str, ground_truth: str):
    if isinstance(ground_truth, str):
        ground_truth = [ground_truth]
    normalized_prediction = normalize_answer(prediction)

    ZERO_METRIC = 0.0
    max_f1 = 0.0

    prediction_tokens = normalized_prediction.split()
    for golden_answer in ground_truth:
        golden_answer_tokens = normalize_answer(golden_answer).split()
        common = Counter(prediction_tokens) & Counter(golden_answer_tokens)
        num_same = sum(common.values())

        if num_same == 0:
            return ZERO_METRIC

        precision = 1.0 * num_same / len(prediction_tokens)
        recall = 1.0 * num_same / len(golden_answer_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        max_f1 = max(f1, max_f1)

    return max_f1

def compute_score_em(solution_str, ground_truth, method="strict", format_score=0.0, score=1.0):
    """The scoring function for exact match (EM).

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """
    answer = extract_solution(solution_str=solution_str)
    score = em_check(answer, ground_truth["target"])
    return score

def compute_format_score(solution: str) -> float:
    ALLOWED_TAGS = {'think', 'tool', 'answer', 'tool_response'}

    def parse_top_level_tags(text: str):
        result = []
        stack = []
        i = 0

        while i < len(text):
            if text[i] != '<':
                i += 1
                continue

            end = text.find('>', i)
            if end == -1:
                i += 1
                continue

            tag_str = text[i + 1:end].strip()

            if tag_str.startswith('/'):
                parts = tag_str[1:].split()
                if not parts:
                    i = end + 1
                    continue
                tag_name = parts[0]
                if stack and stack[-1][0] == tag_name:
                    start_info = stack.pop()
                    if len(stack) == 0:
                        content = text[start_info[2]:i]
                        result.append((tag_name, content))
            elif tag_str and not tag_str.startswith('!'):
                parts = tag_str.split()
                if not parts:
                    i = end + 1
                    continue
                tag_name = parts[0]
                if tag_name and tag_name[0].isalpha():
                    stack.append((tag_name, i, end + 1))

            i = end + 1

        return result

    top_tags = parse_top_level_tags(solution)

    if not top_tags:
        return -0.5

    for tag_name, content in top_tags:
        if tag_name not in ALLOWED_TAGS:
            return -0.5

    state = 'EXPECT_THINK'

    for i, (tag_name, content) in enumerate(top_tags):
        if state == 'EXPECT_THINK':
            if tag_name != 'think':
                return -0.5
            if not content.strip():
                return -0.5
            state = 'EXPECT_TOOL_OR_ANSWER'

        elif state == 'EXPECT_TOOL_OR_ANSWER':
            if tag_name == 'tool':
                if not content.strip():
                    return -0.5
                state = 'EXPECT_TOOL_RESPONSE'
            elif tag_name == 'answer':
                if i != len(top_tags) - 1:
                    return -0.5
                state = 'FINISHED'
            else:
                return -0.5

        elif state == 'EXPECT_TOOL_RESPONSE':
            if tag_name != 'tool_response':
                return -0.5
            state = 'EXPECT_THINK'

        elif state == 'FINISHED':
            return -0.5

    if state != 'FINISHED':
        return -0.5

    return 0.0

def compute_tool_diversity_score(solution_str: str) -> float:
    OPERATORS = {'AnswerGenerate', 'Custom', 'ScEnsemble', 'Refinement', 'Review', 'Revise'}

    tool_response_pattern = r'<tool_response>(.*?)</tool_response>'
    info_pattern = r'<info>(.*?)</info>'

    used_operators = set()

    tool_responses = re.findall(tool_response_pattern, solution_str, re.DOTALL)

    for tr in tool_responses:
        infos = re.findall(info_pattern, tr, re.DOTALL)
        for info in infos:
            for op in OPERATORS:
                if op in info:
                    used_operators.add(op)

    num_tools = len(used_operators)

    if num_tools <= 1:
        return 0.0
    else:
        return min((num_tools - 1) / 4.0, 1.0)

def compute_score(
    solution_str: str,
    ground_truth: Any,
) -> float:
    final_result = extract_solution(solution_str)
    format_score = compute_format_score(solution_str)
    # format_score = 0
    # tool_score = compute_tool_diversity_score(solution_str)
    if final_result:
        equal_result = compute_score_em("<answer>"+final_result+"</answer>", ground_truth)
        score = 1.0 if equal_result else 0.0
    else:
        print("There is no answer block in LLM's response!")
        score = -1.0

    # reward = min(format_score + score + 0.1*tool_score, 1.0)
    reward = format_score + score
    print(f"Workflow: {solution_str}\n\n[DEBUG] Reward: {reward}")
    return reward