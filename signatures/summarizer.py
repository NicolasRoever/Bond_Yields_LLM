import dspy
import config

# Layer 1: Summarize role of the country keyword in the text excerpt
class RoleSummarizer(dspy.Signature):
    """
    ---CONTEXT---
    You are an experienced analyst known for your ability to succinctly summarize complex information and extract key insights from various texts.

    ---TASK---
    Your task is to summarize the role of the country in the given text excerpt.

    ---GUIDELINES---
    - Focus on clearly and concisely capturing the main points related to the country's role.
    - Ensure the summary is accurate and provides a clear understanding of the country's involvement or significance in the text.
    """

    country_keyword = dspy.InputField(desc="keyword that represents a country")
    excerpt = dspy.InputField(desc="excerpt from a financial services company's earnings conference call")
    country_role = dspy.OutputField(desc="one or two sentences")