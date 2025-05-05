import os

import logging
from langchain.agents import Tool, AgentType, initialize_agent
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain.agents import AgentExecutor
from langchain import hub
from langchain.chains import LLMMathChain
from langchain.agents.format_scratchpad import format_log_to_str
from langchain.agents.output_parsers import ReActSingleInputOutputParser
from langchain.tools.render import render_text_description
from langchain_google_genai import ChatGoogleGenerativeAI, HarmBlockThreshold, HarmCategory
import google.generativeai as genai
import json


#use your own API key, from (https://makersuite.google.com/), remember to use VPN if you are not in US
#put your key inside the apostrophe....
os.environ["LANGCHAIN_TRACING"] = "true"
os.environ["LANGSMITH_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGSMITH_API_KEY"] = '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'
os.environ["LANGSMITH_PROJECT"] = "My_project"
os.environ["GOOGLE_API_KEY"] = 'AIzaSyCpLMWak0THJIeAduww7fZM0SePiqTgt2Y'

ddg_search = DuckDuckGoSearchAPIWrapper()

tools = [
    Tool(
        name="DuckDuckGo Search",
        func=ddg_search.run,
        description="Useful to browse information from the internet to know recent results and information you don't know. Then, tell user the result.",
    )
]

# Safety settings
safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
]

# Model name
MODEL_NAME = "gemini-1.5-pro-latest"

# setup model
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro-latest",
    convert_system_message_to_human=True,
    temperature=1,
    top_p=0.95,
    top_k=64,
    max_output_tokens=8192,
    safety_settings = {
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    },
)

# agent_prompt = hub.pull("mikechan/gemini")
# agent_prompt.template
# prompt = "You are a helpful assistant."

# prompt = agent_prompt.partial(
#     tools=render_text_description(tools),
#     tool_names=", ".join([t.name for t in tools]),
# )

# from langchain_core.prompts import ChatPromptTemplate

# memory = ConversationBufferMemory(memory_key="chat_history")

# agent = (
#     {
#         "input": lambda x: x["input"],
#         "agent_scratchpad": lambda x: format_log_to_str(x["intermediate_steps"]),
#         "chat_history": lambda x: x["chat_history"],
#     }
#     | prompt
#     | llm
#     | ReActSingleInputOutputParser()
# )

# prompt = "You are a helpful assistant."

# agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, memory=memory)


# planner_prompt = [
#         (
#             "system",
#             """For the given objective, come up with a simple step by step plan. \
# This plan should involve individual tasks, that if executed correctly will yield the correct answer. Do not add any superfluous steps. \
# The result of the final step should be the final answer. Make sure that each step has all the information needed - do not skip steps.""",
#         ),
#         ("user", "what is the hometown of the current Australia open winner?")
# ]


# Function to send a message to the model
# response = llm.invoke(planner_prompt)

# for chunk in llm.stream(planner_prompt):
#     print(chunk)
from pydantic import BaseModel, Field
from typing import List, Union


class Plan():
    """Plan to follow in future"""

    steps: List[str] = Field(
        description="different steps to follow, should be in sorted order"
    )

from langchain_core.prompts import ChatPromptTemplate

planner_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """For the given objective, come up with a simple step by step plan. \
This plan should involve individual tasks, that if executed correctly will yield the correct answer. Do not add any superfluous steps. \
The result of the final step should be the final answer. Make sure that each step has all the information needed - do not skip steps.""",
        ),
        ("placeholder", "{messages}"),
    ]
)

planner = planner_prompt | llm.with_structured_output(Plan)

planner.invoke(
    {
        "messages": [
            ("user", "what is the hometown of the current Australia open winner?")
        ]
    }
)


class Response():
    """Response to user."""

    response: str


class Act():
    """Action to perform."""

    action: Union[Response, Plan] = Field(
        description="Action to perform. If you want to respond to user, use Response. "
        "If you need to further use tools to get the answer, use Plan."
    )


replanner_prompt = ChatPromptTemplate.from_template(
    """For the given objective, come up with a simple step by step plan. \
This plan should involve individual tasks, that if executed correctly will yield the correct answer. Do not add any superfluous steps. \
The result of the final step should be the final answer. Make sure that each step has all the information needed - do not skip steps.

Your objective was this:
{input}

Your original plan was this:
{plan}

You have currently done the follow steps:
{past_steps}

Update your plan accordingly. If no more steps are needed and you can return to the user, then respond with that. Otherwise, fill out the plan. Only add steps to the plan that still NEED to be done. Do not return previously done steps as part of the plan."""
)

replanner = replanner_prompt | llm.with_structured_output(Act)
