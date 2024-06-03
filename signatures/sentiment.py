import dspy
import config

# Layer 3: Assess sentiment in text excerpts toward solvency of discussed country
scale = [-2, -1, 0, 1, 2]

class SentimentAssessor(dspy.Signature):
    """
    Given the role of the country in the text excerpt, please assess the sentiment towards the solvency of the country. That is the country's financial stability and ability to meet its obligations.
    Please use the following scale for your assessment: -2 = very negative, -1 = somewhat negative, 0 = neutral, 1 = somewhat positive, 2 = very positive
    """

    country_keyword = dspy.InputField(desc="keyword that represents a country")
    country_role = dspy.InputField(desc="role of the country in the excerpt of a financial services company's earnings call transcript")
    sentiment_score = dspy.OutputField(desc=f"one of: {scale}")