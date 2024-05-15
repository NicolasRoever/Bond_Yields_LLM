import os
from dotenv import load_dotenv
import dspy
from dspy import OllamaLocal
from groq import Groq

# Set up API keys
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")

# Set up language models
gpt35turbo = dspy.OpenAI(model = "gpt-3.5-turbo-0125",
                         api_key = openai_api_key,
                         temperature = 0,
                         max_tokens = 800,
                         model_type = "chat")

gpt4turbo = dspy.OpenAI(model = "gpt-4-turbo-2024-04-09",
                         api_key = openai_api_key,
                         temperature = 0,
                         max_tokens = 800,
                         model_type = "chat")

gpt4o = dspy.OpenAI(model = "gpt-4o-2024-05-13",
                         api_key = openai_api_key,
                         temperature = 0,
                         max_tokens = 800,
                         model_type = "chat")

llama3_8b = dspy.OllamaLocal(model = "llama3:8b",
                             temperature = 0,
                             max_tokens = 800)

llama3_70b = dspy.GROQ(model = "llama3-70b-8192",
                       api_key = groq_api_key,
                       temperature = 0,
                       max_tokens = 800)

# Set up default lanugage model to be used when no particular one is specified in the signatures
dspy.settings.configure(lm=llama3_8b)

print(llama3_8b("What is the capital of Italy?"))