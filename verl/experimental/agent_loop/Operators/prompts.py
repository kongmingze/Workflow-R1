ANSWER_GENERATION_PROMPT = """Think step by step to solve the problem.
1. Explain your thinking process in detail inside <think></think> tags.
2. Provide the final answer concisely and clearly inside <answer></answer> tags. The answer should be a direct response to the question, without including explanations or reasoning.

Your response format:
<think>Your detailed step-by-step reasoning here...</think>
<answer>Your final answer here</answer>

Your task: {input}"""

SC_ENSEMBLE_PROMPT = """Given the question described as follows: {question}
Several solutions have been generated to address the given question. They are as follows:
{solutions}

Carefully evaluate these solutions and identify the answer that appears most frequently across them. This consistency in answers is crucial for determining the most reliable solution.

Your response format:
<think>Please Analyze step-by-step the reasoning behind each solution and explain why you choose the most reliable one...</think>
<answer>The best answer directly</answer>"""

REVIEW_PROMPT = """Given a question and a thoughtful solution, your task is to using critical thinking (questioning) to review the solution's correctness and provide a detailed review feedback.

question: {problem}
solution: {solution}

If you are more than 95 percent confident that the final answer is incorrect, please give a feedback which can be utilized to optimize the solution. Otherwise, please give a explanation for the correctness.

Finally, you MUST return the feedback inside <feedback> and </feedback>"""

REVISE_PROMPT = """Given a question and a thoughtful solution which is just reviewed as incorrect, your task is to revise the solution to solve the question.

question: {problem}
solution: {solution}
feedback: {feedback}

You MUST conduct the reasoning process inside <think> and </think>. And then give the revised answer of the question within <answer> and </answer>.

Your response format:
<think>Your step-by-step thinking process here</think>
<answer>Revised answer of the question</answer>"""

REFINEMENT_PROMPT = """Given the query: {query} 
Please try to supply some information and details to make the description of the query more clear, so that help and guide the agent better understand it. Think how to make the clarification more understandable.
Firstly, you MUST conduct the reasoning process inside <think> and </think>. And then give the refined version of the query within <answer> and </answer>.

Your response format:
<think>Your step-by-step thinking process here</think>
<answer>Refined version of the query</answer>"""