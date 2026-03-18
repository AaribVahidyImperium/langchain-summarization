import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import Tool
from langchain.tools import tool

# 1. Setup Environment
load_dotenv()

llm = AzureChatOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
    openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    temperature=0
)

# 2. Define Summarization Tool
summarize_template = "Summarize the following text into exactly 3 sentences: {text}"
summarize_prompt = PromptTemplate.from_template(summarize_template)
summarize_chain = summarize_prompt | llm

def text_summarizer(text: str) -> str:
    """Useful for when you need to summarize a piece of text into a concise 3-sentence summary."""
    response = summarize_chain.invoke({"text": text})
    return response.content

tools = [
    Tool(
        name="TextSummarizer",
        func=text_summarizer,
        description="Summarizes long text into exactly 3 sentences. Input must be a string containing the text to summarize."
    )
]

# 3. ReAct Prompt Template
react_template = """Answer the following questions as best you can. You have access to the following tools:

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

Begin!

Question: {input}
Thought:{agent_scratchpad}"""

prompt = PromptTemplate.from_template(react_template)

# 4. Create the Agent Chain manually to bypass version-dependent imports
# We bind the stop word so the LLM stops generating when it expects an observation.
llm_with_stop = llm.bind(stop=["\nObservation:", "Observation:"])
agent_chain = prompt | llm_with_stop

# 5. Manual Execution Helper (Educational ReAct Loop)
def run_agent(query):
    print(f"\n[Agent invoked with query: {query}]")
    intermediate_steps = ""
    tool_descriptions = "\n".join([f"{t.name}: {t.description}" for t in tools])
    tool_names = ", ".join([t.name for t in tools])
    
    for i in range(5):  # Max 5 iterations
        response = agent_chain.invoke({
            "input": query,
            "agent_scratchpad": intermediate_steps,
            "tools": tool_descriptions,
            "tool_names": tool_names
        })
        
        output = response.content
        print(output)
        
        if "Final Answer:" in output:
            return output.split("Final Answer:")[-1].strip()
            
        if "Action:" in output and "Action Input:" in output:
            action = output.split("Action:")[-1].split("\n")[0].strip()
            action_input = output.split("Action Input:")[-1].split("\n")[0].strip()
            
            tool_to_use = next((t for t in tools if t.name == action), None)
            if tool_to_use:
                try:
                    observation = tool_to_use.invoke(action_input)
                except Exception as e:
                    observation = f"Error: {str(e)}"
            else:
                observation = f"Tool {action} not found."
                
            print(f"Observation: {observation}")
            intermediate_steps += f"{output}\nObservation: {observation}\nThought: "
        else:
            return "Agent failed to format its output correctly."
            
    return "Max iterations reached without a Final Answer."

agent_executor = None

# 6. Test Case 1: Healthcare Impact
healthcare_text = """
Artificial intelligence is revolutionizing healthcare by enhancing diagnostic accuracy and personalized treatment plans. AI algorithms can analyze vast amounts of medical data to identify patterns invisible to humans. This leads to earlier disease detection. AI tools also reduce administrative burdens, allowing providers to focus on patient care. However, integration raises concerns about data security and algorithmic bias.
"""

print("\n" + "="*70)
print("TEST CASE 1: Summarize the impact of AI on healthcare")
print("="*70)

if agent_executor:
    result1 = agent_executor.invoke({"input": f"Summarize the impact of AI on healthcare based on this text: {healthcare_text}"})
    print("\nFinal Summary Result:")
    print(result1["output"])
else:
    output = run_agent(f"Summarize the impact of AI on healthcare based on this text: {healthcare_text}")
    print("\nFinal Summary Result:")
    print(output)

# 7. Test Case 2: Vague Request
print("\n" + "="*70)
print("TEST CASE 2: Vague Request - 'Summarize something interesting'")
print("="*70)

if agent_executor:
    result2 = agent_executor.invoke({"input": "Summarize something interesting."})
    print("\nFinal Result:")
    print(result2["output"])
else:
    output = run_agent("Summarize something interesting.")
    print("\nFinal Result:")
    print(output)