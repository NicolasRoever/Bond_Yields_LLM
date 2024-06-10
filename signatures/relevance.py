import dspy
import config

answer = ["yes", "no"]

class RelevanceAssessor(dspy.Signature):
    """
    ---CONTEXT---
    You are an experienced financial analyst known for your ability to assess and interpret subtle indicators of financial stability and solvency in various countries.

    ---TASK---
    Your task is to assess whether or not the given text excerpt potentially reveals any sentiment towards the solvency of the country mentioned.

    ---GUIDELINES---
    - Focus specifically on potential implications regarding the country's financial stability and ability to meet its obligations.
    - Financial stability and the country's ability to meet its obligations do not need to be discussed explicitly but can be inferred from the text.
    - Answer either 'yes' or 'no'.
    """

    country_keyword = dspy.InputField(desc="keyword that represents a country")
    country_role = dspy.InputField(desc="role of the country in the text excerpt of a finacial services company's earnings call transcript")
    relevance = dspy.OutputField(desc=f"one of: {answer}")