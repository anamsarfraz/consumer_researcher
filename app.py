import os
from dotenv import load_dotenv
import chainlit as cl
import openai
import asyncio
import json
from datetime import datetime

from langsmith.wrappers import wrap_openai
from langsmith import traceable

from article_reader import get_article_content

# Load environment variables
load_dotenv()

configurations = {
    "mistral_7B_instruct": {
        "endpoint_url": os.getenv("MISTRAL_7B_INSTRUCT_ENDPOINT"),
        "api_key": os.getenv("RUNPOD_API_KEY"),
        "model": "mistralai/Mistral-7B-Instruct-v0.3"
    },
    "mistral_7B": {
        "endpoint_url": os.getenv("MISTRAL_7B_ENDPOINT"),
        "api_key": os.getenv("RUNPOD_API_KEY"),
        "model": "mistralai/Mistral-7B-v0.1"
    },
    "openai_gpt-4": {
        "endpoint_url": os.getenv("OPENAI_ENDPOINT"),
        "api_key": os.getenv("OPENAI_API_KEY"),
        "model": "chatgpt-4o-latest"
    }
}

# Choose configuration
config_key = "openai_gpt-4"
#config_key = "mistral_7B_instruct"


# Get selected configuration
config = configurations[config_key]

# Initialize the OpenAI async client
client = wrap_openai(openai.AsyncClient(api_key=config["api_key"], base_url=config["endpoint_url"]))

gen_kwargs = {
    "model": config["model"],
    "temperature": 0.3,
    "max_tokens": 500
}

SYSTEM_PROMPT = """
You are an amazing research specializing in consumer product research. When the user asks for a product suggestion, you must provide a ranked list of top 3 products. For each ranked item, provide product information along with following critreria only:
- cost
- reviews
- pros and cons

Your responses are brief and clear to enabled smooth streaming but should contain enough details to help user make a decision.

Follow guidelines below for generating a response:

1. If the user provides a link, do not mention that you cannot access external links directly. The information in the link will be provided to you as part of user input.
2. If additional ranking criteria is provided, generate top 3 list only with that criteria.
3. If user requests criteria beyond top 3, include upto top 5 products
4. If you are specifying a price mention in USD unless user specifices another currency
5. Mention and use only above criteria unless user specifies additional criteria
"""

# Configuration setting to enable or disable the system prompt
ENABLE_SYSTEM_PROMPT = True
ENABLE_PRODUCT_CONTEXT = True

@traceable
@cl.on_message
async def on_message(message: cl.Message):
    message_history = cl.user_session.get("message_history", [])

    if ENABLE_SYSTEM_PROMPT and (not message_history or message_history[0].get("role") != "system"):
        system_prompt_content = SYSTEM_PROMPT
        if ENABLE_PRODUCT_CONTEXT:
            query = message.content if not message_history else message_history[0].get("content")
            product_context = get_article_content(query, search_google=True)
            system_prompt_content += "\n" + product_context

        message_history.insert(0, {"role": "system", "content": system_prompt_content})

    # if user question contains links, parse the url and get more context
    parsed_message =  get_article_content(message.content) + "\n" + message.content    
    message_history.append({"role": "user", "content": parsed_message})
    
    response_message = cl.Message(content="")
    await response_message.send()

    stream = await client.chat.completions.create(messages=message_history, stream=True, **gen_kwargs)
    async for part in stream:
        if token := part.choices[0].delta.content or "":
            await response_message.stream_token(token)

    message_history.append({"role": "assistant", "content": response_message.content})
    cl.user_session.set("message_history", message_history)
    await response_message.update()


if __name__ == "__main__":
    cl.main()
