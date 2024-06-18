import dspy


class RoleSummarizer(dspy.Signature):
    """
    ---CONTEXT---
    You are an experienced analyst known for your ability to succinctly summarize complex information and extract key insights from various texts.

    ---TASK---
    Your task is to summarize the role of the country in the given text excerpt, i.e. why the country is mentioned.

    ---GUIDELINES---
    - Focus on clearly and concisely capturing the main points related to the country's role.
    - Only include information on one country - the one mentioned in the keyword.
    """

    country_keyword = dspy.InputField(desc="keyword that represents a country")
    excerpt = dspy.InputField(desc="excerpt from a financial services company's earnings conference call")
    answer = dspy.OutputField(desc="one or two sentences")