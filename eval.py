from langsmith.evaluation import evaluate, LangChainStringEvaluator
from langsmith.schemas import Run, Example
from openai import OpenAI
from langsmith import Client
import json
import traceback


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
    Evaluate the model's output on the following two keys metrics:
    - Successful extraction: Model is able to pull out the products with product info (ratings, link, pros / cons, etc)
    - Source quality rating: How legitimate and reliable is this source of the information.
    
    Provide a score from 1 to 4, where 1 is completely non-compliant and 4 is perfectly compliant for both metrics.

Here is the scale you should use to build your answer to score the successful extraction metric with reference to the system_answer:
1: The information_extraction is terrible: completely irrelevant to the question asked, or very partial
2: The information_extraction is mostly not helpful: misses some key aspects of the question
3: The information_extraction is mostly helpful: provides good information, but still could be improved
4: The information_extraction is excellent: relevant, direct, detailed, and addresses all the aspects of the question

Here is the scale you should use to build your answer to score the quality of the source for being reliable and legitimate:
1: The source_quality is not reliable at all: completely irrelevant to the question asked, or very partial. The answer cannot be generated using this information.
2: The source_quality is mostly not reliable: misses some key aspects of the question and doesn't contain enough data to generate answers
3: The source_quality is mostly reliable: provides helpful information to generate answers, but still could be improved
4: The source_quality is highly reliable: relevant, direct, detailed, and contains all the neccessary information to generate a good answer.

    Respond in the following correct JSON format with double quotes:
    {[
        {
            "key": "information_extraction",
            "score": "<int>",
            "explanation": "<string>"
        },
        {
            "key": "source_quality",
            "score": "<int>",
            "explanation": "<string>"
        }
    ]}
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are an AI assistant tasked with evaluating the compliance of model outputs to given prompts and conversation context."},
            {"role": "user", "content": evaluation_prompt}
        ],
        temperature=0.2
    )

    try:
        res_content = response.choices[0].message.content

        print("Response: {}".format(res_content))
        result = json.loads(res_content[res_content.find('['):])
        return {
            "results": [{
                "key": result[0]["key"],
                "score": int(result[0]["score"]) /4,
                "reason": result[0]["explanation"]
            },
            {
                "key": result[1]["key"],
                "score": int(result[1]["score"]) /4,
                "reason": result[1]["explanation"]
            }]
        }

    except json.JSONDecodeError:
        return {
            "results": [{
                "key": "information_extraction",
                "score": 0,
                "reason": "Failed to parse evaluator response: {}\nEvaluator response: {}".format(traceback.format_exc(), response.choices[0].message.content)
            },
            {
                "key": "source_quality",
                "score": 0,
                "reason": "Failed to parse evaluator response"
            }]
        }


# Langsmit Client
lang_client = Client()
# The name or UUID of the LangSmith dataset to evaluate on.
data_set =  "consumer_researcher_ds"
# A string to prefix the experiment name with.
experiment_prefix = "cr_metric_compliance"

# List of evaluators to score the outputs of target task
evaluators = [
    prompt_compliance_evaluator
]

# Evaluate the target task
results = evaluate(
    lambda inputs: inputs,
    data=lang_client.list_examples(dataset_name=data_set, splits=["base"]),
    evaluators=evaluators,
    experiment_prefix=experiment_prefix,
)
