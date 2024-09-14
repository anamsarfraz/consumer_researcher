from langsmith.evaluation import evaluate, LangChainStringEvaluator
from langsmith.schemas import Run, Example
from openai import OpenAI
import json

from dotenv import load_dotenv
load_dotenv()

from langsmith.wrappers import wrap_openai
from langsmith import traceable

client = wrap_openai(OpenAI())

@traceable
def prompt_compliance_evaluator(run: Run, example: Example) -> dict:
    inputs = example.inputs['input']
    outputs = example.outputs['output']

    # Extract system prompt
    system_prompt = next((msg['data']['content'] for msg in inputs if msg['type'] == 'system'), "")

    # Extract message history
    message_history = []
    for msg in inputs:
        if msg['type'] in ['human', 'ai']:
            message_history.append({
                "role": "user" if msg['type'] == 'human' else "assistant",
                "content": msg['data']['content']
            })

    # Extract latest user message and model output
    latest_message = message_history[-1]['content'] if message_history else ""
    model_output = outputs['data']['content']

    evaluation_prompt = f"""
    System Prompt: {system_prompt}

    Message History:
    {json.dumps(message_history, indent=2)}

    Latest User Message: {latest_message}

    Model Output: {model_output}

    Based on the above information, evaluate the model's output for compliance with the system prompt and context of the conversation. 
    Provide a score from 1 to 4, where 1 is completely non-compliant and 4 is perfectly compliant.

Here is the scale you should use to build your answer to score the system_answer:
1: The system_answer is terrible: completely irrelevant to the question asked, or very partial
2: The system_answer is mostly not helpful: misses some key aspects of the question
3: The system_answer is mostly helpful: provides good information, but still could be improved
4: The system_answer is excellent: relevant, direct, detailed, and addresses all the concerns raised in the question

Here is the scale you should use to build your answer to score the quality of the original source ini terms of reliability from where the context has been provided:
1: The source_quality is not reliable at all: completely irrelevant to the question asked, or very partial
2: The source_quality is mostly not reliable: misses some key aspects of the question
3: The source_quality is mostly reliable: provides good information, but still could be improved
4: The source_quality is highly reliable: relevant, direct, detailed, and addresses all the concerns raised in the question

You can give floating point score as well with 1-4
    Respond in the following JSON format:
    {{
        "source_quality": <int>,
        "score": <int>
        "explanation": "<string>"
    }}
    """

    response = client.chat.completions.create(
        model="chatgpt-4o-latest",
        messages=[
            {"role": "system", "content": "You are an AI assistant tasked with evaluating the compliance of model outputs to given prompts and conversation context."},
            {"role": "user", "content": evaluation_prompt}
        ],
        temperature=0.2
    )

    try:
        result = json.loads(response.choices[0].message.content)
        return {
            "key": "prompt_compliance",
            "successful_information_extraction": result["score"] / 4,  # Normalize to 0-1 range
            "score": result["source_quality"] /4,
            "reason": result["explanation"]
        }
    except json.JSONDecodeError:
        return {
            "key": "prompt_compliance",
            "score": 0,
            "reason": "Failed to parse evaluator response"
        }


# The name or UUID of the LangSmith dataset to evaluate on.
data_set =  "consumer_researcher_ds"
# A string to prefix the experiment name with.
experiment_prefix = "cr_prompt_compliance"

# List of evaluators to score the outputs of target task
evaluators = [
    prompt_compliance_evaluator
]

# Evaluate the target task
results = evaluate(
    lambda inputs: inputs,
    data=data_set,
    evaluators=evaluators,
    experiment_prefix=experiment_prefix,
)

print(results)