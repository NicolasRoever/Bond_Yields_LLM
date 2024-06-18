import dspy

answer = ["yes", "no"]

class RelevanceAssessor(dspy.Signature):
    """
    ---CONTEXT---
    You are an experienced financial analyst known for your ability to assess and interpret subtle indicators of financial stability and solvency in various countries.

    ---TASK---
    Your task is to assess whether or not the given text excerpt potentially reveals any expectation towards the solvency of the country mentioned.

    ---GUIDELINES---
    - Focus specifically on potential implications regarding the country's financial stability and ability to meet its obligations.
    - Financial stability and the country's ability to meet its obligations do not need to be discussed explicitly but can be inferred from the text.
    - Answer either 'yes' for relevant or 'no' for irrelevant.
    """

    country_keyword = dspy.InputField(desc="keyword that represents a country")
    country_role = dspy.InputField(desc="role of the country in the text excerpt of a finacial services company's earnings call transcript")
    answer = dspy.OutputField(desc=f"one of: {answer}")



def extract_relevance_as_yes_or_no(text):
    """"Extracts the relevance assessment from the text and returns 'yes' or 'no'"""
    if 'Answer: yes' in text:
        return 'yes'
    elif 'Answer: no' in text:
        return 'no'
    else:
        return 'NA'