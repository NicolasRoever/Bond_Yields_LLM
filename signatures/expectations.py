import dspy

# Layer 3: Assess sentiment in text excerpts toward solvency of discussed country
scale = [-2, -1, 0, 1, 2]

class ExpectationAssessor(dspy.Signature):
    """
    ---CONTEXT---
    You are an experienced financial analyst known for your expertise in evaluating and interpreting expectations related to the financial stability and solvency of countries.

    ---TASK---
    Please assess the expectation towards the solvency of the country mentioned in the given text excerpt. This includes evaluating the country's financial stability and ability to meet its obligations.    
    
    ---GUIDELINES---
    - Use the following scale for your assessment: -2 = very negative, -1 = somewhat negative, 0 = neutral, 1 = somewhat positive, 2 = very positive
    """

    country_keyword = dspy.InputField(desc="keyword that represents a country")
    country_role = dspy.InputField(desc="role of the country in the excerpt of a financial services company's earnings call transcript")
    excerpt = dspy.InputField(desc="excerpt from a financial services company's earnings conference call")
    answer = dspy.OutputField(desc=f"one of: {scale}. Only respond wiht a single digit and nothing else. Don't include any prefix (e.g. 'Sentiment: ').", format=str)