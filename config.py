import os
from dotenv import load_dotenv
import dspy
from dsp.modules.anthropic import Claude
from dspy import OllamaLocal
from groq import Groq

# Set up API keys
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
#groq_api_key = os.getenv("GROQ_API_KEY")
#anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")

# Set up supported language models
gpt35turbo = dspy.OpenAI(model = "gpt-3.5-turbo-0125",
                         api_key = openai_api_key,
                         temperature = 0,
                         max_tokens = 800,
                         model_type = "chat")

# gpt4turbo = dspy.OpenAI(model = "gpt-4-turbo-2024-04-09",
#                          api_key = openai_api_key,
#                          temperature = 0,
#                          max_tokens = 800,
#                          model_type = "chat")

# gpt4o = dspy.OpenAI(model = "gpt-4o-2024-05-13",
#                          api_key = openai_api_key,
#                          temperature = 0,
#                          max_tokens = 800,
#                          model_type = "chat")

# haiku = Claude(model = "claude-3-haiku-20240307",
#                api_key = anthropic_api_key,
#                temperature = 0,
#                max_tokens = 800)

llama3_8b = dspy.OllamaLocal(model = "llama3:8b",
                             temperature = 0,
                             max_tokens = 800)

# llama3_70b = dspy.GROQ(model = "llama3-70b-8192",
#                        api_key = groq_api_key,
#                        temperature = 0,
#                        max_tokens = 800)

# Set up default lanugage model to be used when no particular one is specified in the signatures
dspy.settings.configure(lm=llama3_8b)


def convert_to_integer_if_answer_is_valid(value):
    """
    Converts a string to an integer if it matches one of the specified values:
    "-2", "-1", "0", "1", "2", or "99". Otherwise, returns the original string.

    Parameters:
    value (str): The input string to be converted.

    Returns:
    int or str: The integer value if the input matches one of the specified strings,
                otherwise the original string.
    """
    return int(value) if value in {"-2", "-1", "0", "1", "2", "99"} else value

def check_if_answer_is_among_valid_answers(value):
    """
    Returns true if string matches one of the specified values:
    "-2", "-1", "0", "1", "2", or "99". Otherwise, returns False

    Parameters:
    value (str): The input string to be checked.

    Returns:
    bool: True or False
    """
    return True if value in {"-2", "-1", "0", "1", "2", "99"} else False