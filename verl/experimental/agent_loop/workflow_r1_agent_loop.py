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
import asyncio
import json
import logging
import os
from typing import Any
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, register
from verl.experimental.agent_loop.tool_parser import FunctionCall, ToolParser
from verl.experimental.agent_loop.Operators.operators_tool import Custom, AnswerGenerate, ScEnsemble, WarningOp, PromptOptimizer, Review, Revise, MultiAnswerGenerate, Search
from verl.utils.profiler import simple_timer
from verl.utils.rollout_trace import rollout_trace_op

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

OPERATORS = {
    "Custom": Custom,
    "AnswerGenerate": AnswerGenerate,
    "ScEnsemble": ScEnsemble,
    "WarningOp": WarningOp,
    "PromptOptimizer": PromptOptimizer,
    "Review": Review,
    "Revise": Revise,
    "MultiAnswerGenerate": MultiAnswerGenerate,
    "Search": Search,
}


@register("workflow_r1_agent_loop")
class WorkflowR1AgentLoop(AgentLoopBase):
    @classmethod
    def init_class(cls, config, tokenizer, **kwargs):
        if cls._class_initialized:
            return
        cls._class_initialized = True
        print("Performing class-level WorkflowR1AgentLoop initialization")

        cls.tokenizer = tokenizer
        cls.max_user_turns = config.actor_rollout_ref.rollout.multi_turn.max_user_turns
        cls.max_assistant_turns = config.actor_rollout_ref.rollout.multi_turn.max_assistant_turns
        cls.max_parallel_calls = config.actor_rollout_ref.rollout.multi_turn.max_parallel_calls
        cls.max_tool_response_length = config.actor_rollout_ref.rollout.multi_turn.max_tool_response_length
        cls.tool_response_truncate_side = config.actor_rollout_ref.rollout.multi_turn.tool_response_truncate_side

        cls.tool_parser = ToolParser.get_tool_parser(
            config.actor_rollout_ref.rollout.multi_turn.format, cls.tokenizer
        )
        print(f"Initialized WorkflowR1AgentLoop with operators: {list(OPERATORS.keys())}")

        cls.prompt_length = config.actor_rollout_ref.rollout.prompt_length
        cls.response_length = config.actor_rollout_ref.rollout.response_length
        cls.system_prompt = tokenizer.apply_chat_template([{}], add_generation_prompt=False, tokenize=True)

    @rollout_trace_op
    async def run(self, messages: list[dict[str, Any]], sampling_params: dict[str, Any]) -> AgentLoopOutput:
        metrics = {}
        request_id = uuid4().hex
        prompt_ids = await self.loop.run_in_executor(
            None,
            lambda: self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True
            ),
        )
        response_mask = []

        user_turns, assistant_turns = 0, 0
        while True:
            with simple_timer("generate_sequences", metrics):
                response_ids = await self.server_manager.generate(
                    request_id=request_id, prompt_ids=prompt_ids, sampling_params=sampling_params
                )
            prompt_ids += response_ids
            response_mask += [1] * len(response_ids)
            assistant_turns += 1

            if len(response_mask) >= self.response_length:
                break

            if self.max_assistant_turns and assistant_turns >= self.max_assistant_turns:
                break

            if self.max_user_turns and user_turns >= self.max_user_turns:
                break

            _, tool_calls = await self.tool_parser.extract_tool_calls(response_ids)
            if not tool_calls:
                break

            tasks = []
            for tool_call in tool_calls[:self.max_parallel_calls]:
                tasks.append(self._call_tool(tool_call))
            with simple_timer("tool_calls", metrics):
                tool_responses = await asyncio.gather(*tasks)
            await asyncio.sleep(1)
            if any(isinstance(item, Exception) for item in tool_responses):
                break

            tool_response_ids = await self.loop.run_in_executor(
                None,
                lambda messages=tool_responses: self.tokenizer.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=True
                ),
            )
            tool_response_ids = tool_response_ids[len(self.system_prompt):]

            if len(response_mask) + len(tool_response_ids) >= self.response_length:
                break

            prompt_ids += tool_response_ids
            response_mask += [0] * len(tool_response_ids)
            user_turns += 1

        response_ids = prompt_ids[-len(response_mask):]
        prompt_ids = prompt_ids[:len(prompt_ids) - len(response_mask)]

        output = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids[:self.response_length],
            response_mask=response_mask[:self.response_length],
            num_turns=user_turns + assistant_turns + 1,
            metrics=metrics,
        )
        return output

    async def _call_tool(self, tool_call: FunctionCall) -> dict[str, str]:
        try:
            operator_name = tool_call.name
            args = json.loads(tool_call.arguments)
            content = args.get("content", "")

            if operator_name not in OPERATORS:
                tool_response = f"Unknown operator: {operator_name}. Available operators: {list(OPERATORS.keys())}"
            else:
                operator_fn = OPERATORS[operator_name]
                if operator_name == "WarningOp":
                    tool_response = await operator_fn()
                else:
                    tool_response = await operator_fn(content)

        except Exception as e:
            logger.exception(f"Error when executing operator: {e}")
            return e

        if len(tool_response) > self.max_tool_response_length:
            if self.tool_response_truncate_side == "left":
                tool_response = tool_response[:self.max_tool_response_length] + "...(truncated)"
            elif self.tool_response_truncate_side == "right":
                tool_response = "(truncated)..." + tool_response[-self.max_tool_response_length:]
            else:
                length = self.max_tool_response_length // 2
                tool_response = tool_response[:length] + "...(truncated)..." + tool_response[-length:]

        return {
            "role": "tool",
            "content": f"<info>{tool_response}</info>",
        }

