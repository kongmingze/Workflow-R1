import asyncio
import aiohttp
import re
import os
from typing import List
import json
import ssl
import certifi

BASE_URL = os.getenv("OPERATOR_BASE_URL", "")
API_KEY = os.getenv("OPERATOR_API_KEY", "")
MODEL_NAME = os.getenv("OPERATOR_MODEL_NAME", "")
SEARCH_API_KEY = os.getenv("SEARCH_API_KEY", "")
SEARCH_URL = os.getenv("SEARCH_URL", "")

from .prompts import ANSWER_GENERATION_PROMPT, SC_ENSEMBLE_PROMPT, REFINEMENT_PROMPT, REVISE_PROMPT, REVIEW_PROMPT

MAX_NEW_TOKENS = 1024
TIMEOUT = 600

def parse_tag(content: str, tag: str) -> str:
    pattern = rf'<{tag}>(.*?)</{tag}>'
    match = re.search(pattern, content, re.DOTALL)
    return match.group(1).strip() if match else ""

def parse_tags(content: str, tag: str) -> List[str]:
    pattern = rf'<{tag}>(.*?)</{tag}>'
    matches = re.findall(pattern, content, re.DOTALL)
    return [m.strip() for m in matches]

async def llm_request_single(messages: List[dict]) -> tuple:
    url = f"{BASE_URL}/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": 0,
        "max_tokens": MAX_NEW_TOKENS
    }

    try:
        timeout = aiohttp.ClientTimeout(total=TIMEOUT)
        ssl_context = ssl.create_default_context(cafile=certifi.where())
        connector = aiohttp.TCPConnector(ssl=ssl_context)
        async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
            async with session.post(url, json=payload, headers=headers) as response:
                if response.status != 200:
                    error_text = await response.text()
                    print(f"LLM request error: {response.status} - {error_text}")
                    return "", False, "HTTPError"

                result = await response.json()
                choice = result["choices"][0]
                content = choice["message"]["content"]
                finish_reason = choice.get("finish_reason", "")
                return content, finish_reason == "length", None

    except asyncio.TimeoutError:
        return "", False, "TimeoutError"
    except Exception as e:
        print(f"LLM request error: {e}")
        return "", False, "OtherError"


async def llm_request_with_retry(prompt: str, required_tags: List[str], max_retries: int = 3) -> str:
    messages = [{"role": "user", "content": prompt}]
    last_response = ""

    for attempt in range(max_retries):
        response, was_truncated, error = await llm_request_single(messages)
        if error:
            continue

        last_response = response
        missing_tags = [tag for tag in required_tags if not parse_tag(response, tag)]

        if not missing_tags and not was_truncated:
            return response

        if attempt < max_retries - 1:
            messages.append({"role": "assistant", "content": response})
            hints = []
            if was_truncated:
                hints.append("Your response was TRUNCATED. Please be MORE CONCISE.")
            if missing_tags:
                hints.append(f"MISSING required tags: {missing_tags}. Include ALL required tags.")
            messages.append({"role": "user", "content": f"{' '.join(hints)}\n\n{prompt}"})

    return last_response

def format_with_think(operator_name: str, content: str, thought_process=True) -> str:
    think, answer = parse_tag(content, "think"), parse_tag(content, "answer")
    if think and answer and thought_process:
        return f"{operator_name} thought process: {think}\n\n{operator_name} answer: {answer}"
    return f"{operator_name} answer: {answer}" if answer else content

def format_with_raw(operator_name: str, content: str) -> str:
    answer = parse_tag(content, "answer")
    return f"{operator_name} answer: {answer}" if answer else content

async def Custom(content: str) -> str:
    task, instruction = parse_tag(content, "task"), parse_tag(content, "instruction")
    if not task and not instruction:
        return "You did not follow the required format. If I want to use Custom operator correctly, I need to rethink and provide the task description within <task></task> tag and instruction within <instruction></instruction> tag."
    prompt = f"{instruction}\n\n{task}\n\nFinally, you should put your final answer within <answer></answer> tags."
    return format_with_raw("Custom", await llm_request_with_retry(prompt, ["answer"]))

async def AnswerGenerate(content: str) -> str:
    prompt = ANSWER_GENERATION_PROMPT.format(input=content)
    return format_with_think("AnswerGenerate", await llm_request_with_retry(prompt, ["think", "answer"]))

async def MultiAnswerGenerate(content: str) -> str:
    prompt = ANSWER_GENERATION_PROMPT.format(input=content)
    results = []
    for i in range(3):
        response = await llm_request_with_retry(prompt, ["think", "answer"])
        results.append(f"[Answer {i+1}]\n{format_with_think('AnswerGenerate', response, thought_process=False)}")
    return "\n\n".join(results)

async def ScEnsemble(content: str) -> str:
    query, solutions = parse_tag(content, "query"), parse_tags(content, "solution")
    if not query:
        return "You did not follow the required format. If I want to use ScEnsemble operator correctly, I need to provide the query within <query></query> tag."
    if len(solutions) == 0:
        return "You did not follow the required format. If I want to use ScEnsemble operator correctly, I need to provide solutions within <solution></solution> tags."
    if len(solutions) == 1:
        return "Using ScEnsemble with only one solution is meaningless. To make ensemble selection effective, I need to provide multiple candidate answers for this query, each wrapped within separate <solution></solution> tags."

    solution_text = "".join(f"{chr(65 + i)}: {sol}\n\n" for i, sol in enumerate(solutions))
    prompt = SC_ENSEMBLE_PROMPT.format(question=query, solutions=solution_text)
    return format_with_think("ScEnsemble", await llm_request_with_retry(prompt, ["think", "answer"]))

async def WarningOp() -> str:
    return "WARNING!!! You did not follow the required format. If you want to use a tool, you should use <tool>OperatorName: input</tool>. If you want to give the final answer, you should use <answer>...</answer>."

async def PromptOptimizer(content: str) -> str:
    prompt = REFINEMENT_PROMPT.format(query=content)
    return format_with_think("PromptOptimizer", await llm_request_with_retry(prompt, ["think", "answer"]))


async def Review(content: str) -> str:
    question = parse_tag(content, "question")
    solution = parse_tag(content, "solution")
    
    if not question:
        return "You did not follow the required format. If I want to use Review operator correctly, I need to provide the question within <question></question> tag."
    if not solution:
        return "You did not follow the required format. If I want to use Review operator correctly, I need to provide the solution within <solution></solution> tag."
    
    prompt = REVIEW_PROMPT.format(problem=question, solution=solution)
    response = await llm_request_with_retry(prompt, ["feedback"])
    feedback = parse_tag(response, "feedback")
    return f"Review feedback: {feedback}" if feedback else response


async def Revise(content: str) -> str:
    question = parse_tag(content, "question")
    solution = parse_tag(content, "solution")
    feedback = parse_tag(content, "feedback")
    
    if not question:
        return "You did not follow the required format. If I want to use Revise operator correctly, I need to provide the problem within <question></question> tag."
    if not solution:
        return "You did not follow the required format. If I want to use Revise operator correctly, I need to provide the solution within <solution></solution> tag."
    if not feedback:
        return "You did not follow the required format. If I want to use Revise operator correctly, I need to provide the feedback within <feedback></feedback> tag."
    
    prompt = REVISE_PROMPT.format(problem=question, solution=solution, feedback=feedback)
    return format_with_think("Revise", await llm_request_with_retry(prompt, ["think", "answer"]))

async def Search(content: str) -> str:
    query = content
    max_retries = 3
    max_results = 5
    
    payload = json.dumps({"q": query, "gl": "cn", "num": max_results})
    headers = {
        "X-API-KEY": SEARCH_API_KEY,
        "Content-Type": "application/json"
    }
    
    last_error = None
    for attempt in range(max_retries):
        try:
            timeout = aiohttp.ClientTimeout(total=60)
            ssl_context = ssl.create_default_context(cafile=certifi.where())
            connector = aiohttp.TCPConnector(ssl=ssl_context)
            
            async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
                async with session.post(SEARCH_URL, data=payload, headers=headers) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        return f"Search error: {response.status} - {error_text}"
                    
                    result = await response.json()

            snippets = []
            for i, item in enumerate(result.get("organic", [])[:max_results], 1):
                title = item.get("title", "")
                snippet = item.get("snippet", "")
                snippets.append(f"Title: {title}\nSnippet: {snippet}")
            
            if not snippets:
                return f"Search: No results found for query: {query}"
            
            search_results = "\n\n".join(snippets)
            return f"Search results for '{query}':\n\n{search_results}"
        
        except asyncio.TimeoutError:
            last_error = "Request timed out"
        except Exception as e:
            last_error = str(e)
        
        if attempt < max_retries - 1:
            await asyncio.sleep(5)
    
    return f"Search error after {max_retries} retries: {last_error}"