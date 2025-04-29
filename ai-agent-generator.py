import os

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

#use your own API key, from (https://makersuite.google.com/), remember to use VPN if you are not in US
#put your key inside the apostrophe....
os.environ["LANGCHAIN_TRACING"] = "true"
os.environ["LANGSMITH_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGSMITH_API_KEY"] = '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'
os.environ["LANGSMITH_PROJECT"] = "My_project"
os.environ["GOOGLE_API_KEY"] = 'AIzaSyCpLMWak0THJIeAduww7fZM0SePiqTgt2Y'


# setup model
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro-latest",
    convert_system_message_to_human=True,
    handle_parsing_errors=True,
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

ddg_search = DuckDuckGoSearchAPIWrapper()
llm_math_chain = LLMMathChain.from_llm(llm=llm, verbose=True)

tools = [
    Tool(
        name="DuckDuckGo Search",
        func=ddg_search.run,
        description="Useful to browse information from the internet to know recent results and information you don't know. Then, tell user the result.",
    ),
    Tool.from_function(
        func=llm_math_chain.run,
        name="Calculator",
        description="Useful for when you need to answer questions about very simple math. This tool is only for math questions and nothing else. Only input math expressions. Not supporting symbolic math.",
    ),
]

agent_prompt = hub.pull("mikechan/gemini")
agent_prompt.template

prompt = agent_prompt.partial(
    tools=render_text_description(tools),
    tool_names=", ".join([t.name for t in tools]),
)
llm_with_stop = llm.bind(stop=["\nObservation"])

memory = ConversationBufferMemory(memory_key="chat_history")

agent = (
    {
        "input": lambda x: x["input"],
        "agent_scratchpad": lambda x: format_log_to_str(x["intermediate_steps"]),
        "chat_history": lambda x: x["chat_history"],
    }
    | prompt
    | llm_with_stop
    | ReActSingleInputOutputParser()
)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, memory=memory)

agent_executor.invoke({"input": "calculate this: The marked price of a chair is 20% higher than its cost. The chair is sold at a discount of $90 on its marked price. After selling the chair, the percentage loss is 16%. Find the cost of the chair."})["output"]
