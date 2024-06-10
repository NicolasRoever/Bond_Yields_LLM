import dspy
import config

# Layer 3: Assess sentiment in text excerpts toward solvency of discussed country
scale = [-2, -1, 0, 1, 2]

class SentimentAssessor(dspy.Signature):
    """
    ---CONTEXT---
    You are an experienced financial analyst known for your expertise in evaluating and interpreting sentiments related to the financial stability and solvency of countries.

    ---TASK---
    Please assess the sentiment towards the solvency of the country mentioned in the given text excerpt. This includes evaluating the country's financial stability and ability to meet its obligations.    
    
    ---GUIDELINES---
    - Use the following scale for your assessment: -2 = very negative, -1 = somewhat negative, 0 = neutral, 1 = somewhat positive, 2 = very positive
    """

    country_keyword = dspy.InputField(desc="keyword that represents a country")
    country_role = dspy.InputField(desc="role of the country in the excerpt of a financial services company's earnings call transcript")
    sentiment_score = dspy.OutputField(desc=f"one of: {scale}. Only respond wiht a single digit and nothing else. Don't include any prefix (e.g. 'Sentiment: ').")