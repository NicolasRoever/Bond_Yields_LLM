import dspy
import config

answer = ["yes", "no"]

class RelevanceAssessor(dspy.Signature):
    """
    Given the role of the country in the text excerpt, your task is to assess whether or not the excerpt potentially reveals any sentiment towards the solvency of the country.
    Focus specifically on potential implications regarding the country's financial stability and ability to meet its obligations.
    Financial stability and the country's ability to meet its obligations do not need to be discussed explicitly, but can be inferred from the text.
    Answer either 'yes' or 'no'.
    """

    country_keyword = dspy.InputField(desc="keyword that represents a country")
    country_role = dspy.InputField(desc="role of the country in the text excerpt of a finacial services company's earnings call transcript")
    relevance = dspy.OutputField(desc=f"one of: {answer}")