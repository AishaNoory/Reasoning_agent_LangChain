#import libraries and modules
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser, agent
from langchain.chains import llm
from langchain.prompts import StringPromptTemplate
from langchain import OpenAI, LLMChain
from typing import List, Union
import re
from langchain.schema import OutputParserException
import openai
import requests
from bs4 import BeautifulSoup, TemplateString
import gradio as gr
# Set the OpenAI API key
openai.api_key = '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'

# Set up tools
# Define which tools the agent can use to answer user queries
def perform_search(query):
    search_url = f'https://www.google.com/search?q={query}'
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
    response = requests.get(search_url, headers=headers)
    
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        search_results = soup.find_all('div', class_='tF2Cxc')
        results = []
        for result in search_results:
            title = result.find('h3', class_='LC20lb').text
            link = result.find('a')['href']
            results.append({'title': title, 'link': link})
        return results
    else:
        return []

tools = [
    Tool(
        name="Search",
        func=perform_search,
        description="useful for when you need to answer questions about current events"
    )
]

# Set up the base template
template = """Answer the following questions as best you can, but speaking as a pirate might speak. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin! Remember to speak as a pirate when giving your final answer. Use lots of "Arg"s

Question: {input}
{agent_scratchpad}"""

# Set up a prompt template
class CustomPromptTemplate(StringPromptTemplate):
    # The template to use
    template: str
    # The list of tools available
    tools: List[Tool]

    def format(self, **kwargs) -> str:
        # Get the intermediate steps (AgentAction, Observation tuples)
        # Format them in a particular way
        intermediate_steps = kwargs.pop("intermediate_steps")
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\nObservation: {observation}\nThought: "
        # Set the agent_scratchpad variable to that value
        kwargs["agent_scratchpad"] = thoughts
        # Create a tools variable from the list of tools provided
        kwargs["tools"] = "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])
        # Create a list of tool names for the tools provided
        kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])
        return self.template.format(**kwargs)

# Define a function to parse LLMSingleActionAgent output
class CustomOutputParser(AgentOutputParser):
    # ...

# Create a list of tools
 tools = [
    Tool(
        name="Search",
        func=perform_search,
        description="useful for when you need to answer questions about current events"
    )
]

# Define your CustomPromptTemplate and use the 'tools' variable
prompt = CustomPromptTemplate(
    template=template,
    tools=tools,
    # This omits the `agent_scratchpad`, `tools`, and `tool_names` variables because those are generated dynamically
    # This includes the `intermediate_steps` variable because that is needed
    input_variables=["input", "intermediate_steps"]
)

# Create an instance of the GPT-3 chat model using openai.ChatCompletion.create()
response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You are a helpful assistant that provides information about current events."},
        {"role": "user", "content": "What happened in the world today?"}
    ],
    temperature=0
)

# Get the assistant's reply from the response
assistant_reply = response.choices[0].message['content']
print(assistant_reply)

# Set up the LLM chain with prompt template
prompt_with_history = CustomPromptTemplate(
    template=template,  # Use the template_with_history you defined earlier
    tools=tools,
    input_variables=["input", "intermediate_steps", "history"]
)
llm_chain = LLMChain(llm=llm, prompt=prompt_with_history)
tool_names = [tool.name for tool in tools]
agent = LLMSingleActionAgent(
    llm_chain=llm_chain,
    output_parser=OutputParserException,
    stop=["\nObservation:"],
    allowed_tools=tool_names
)
from langchain.memory import ConversationBufferWindowMemory
memory=ConversationBufferWindowMemory(k=2)

# Create an AgentExecutor and use the agent
agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True, memory=memory)
agent_executor.run("How many people live in Canada as of 2023?")


