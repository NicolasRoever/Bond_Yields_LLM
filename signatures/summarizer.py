import dspy
import config

# Layer 1: Summarize role of the country keyword in the text excerpt
class RoleSummarizer(dspy.Signature):
    """
    Your task is to summarize the role of the country in the given text excerpt.
    """

    country_keyword = dspy.InputField(desc="keyword that represents a country")
    excerpt = dspy.InputField(desc="excerpt from a financial services company's earnings conference call")
    country_role = dspy.OutputField(desc="one or two sentences")