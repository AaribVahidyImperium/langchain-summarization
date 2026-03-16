import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
# Updated modern import
from langchain_core.prompts import PromptTemplate 

# 1. Load environment variables
load_dotenv()

# 2. Configure the Azure OpenAI Model
llm = AzureChatOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
    openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    temperature=0
)

# 3. Design a Prompt Template
template = "Summarize the following text into exactly 3 sentences: {text}"
prompt = PromptTemplate.from_template(template)

# 4. Create the Chain (LCEL)
chain = prompt | llm

print("Model and Chain are ready.")



# 5. Test Data (Approx 200 words about AI)
ai_paragraph = """
Artificial Intelligence (AI) refers to the simulation of human intelligence in machines that are programmed to think like humans and mimic their actions. The term may also be applied to any machine that exhibits traits associated with a human mind such as learning and problem-solving. As technology advances, previous benchmarks that defined intelligence become outdated. For example, machines that calculate basic functions or recognize text through optical character recognition are no longer considered to embody artificial intelligence, since this function is now taken for granted as an inherent computer function. AI is continuously evolving to benefit many different industries. Machines are wired using a cross-disciplinary approach based on mathematics, computer science, linguistics, psychology, and more. Algorithms often play a very important part in the structure of artificial intelligence, where simple algorithms are used in simple applications, while more complex ones help frame strong artificial intelligence. Beyond just automation, modern AI systems are now capable of creative tasks, complex decision-making in healthcare, and even driving autonomous vehicles. This rapid growth has sparked global discussions regarding the ethical implications of AI, including data privacy, job displacement, and the potential for algorithmic bias in critical systems.
"""

# 6. Run the 3-sentence summary
print("\n--- Generating 3-Sentence Summary ---")
response_3 = chain.invoke({"text": ai_paragraph})
print(response_3.content)



# 7.Creating a new Prompt for 1 sentence
template_1_sentence = "Summarize the following text into exactly 1 sentence: {text}"
prompt_1 = PromptTemplate.from_template(template_1_sentence)

# 8. Creating a new Chain for the 1-sentence prompt
chain_1 = prompt_1 | llm

# 9. Running the 1-sentence summary
print("\n--- Generating 1-Sentence Summary ---")
response_1 = chain_1.invoke({"text": ai_paragraph})
print(response_1.content)