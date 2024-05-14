import dspy
import config

class RelevanceAssessor(dspy.Signature):
    """
    Given the role of the country in the text excerpt, your task is to assess whether or not the excerpt potentially reveals any sentiment towards the solvency of the country.
    """

    country_keyword = dspy.InputField(desc="keyword that represents a country")
    country_role = dspy.InputField(desc="role of the country in the text excerpt")
    relevance = dspy.OutputField(desc="either yes or no")