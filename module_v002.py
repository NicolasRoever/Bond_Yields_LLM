
import dspy
from signatures.summarizer import RoleSummarizer
from signatures.relevance import RelevanceAssessor
from signatures.expectations import ExpectationAssessor
from config import convert_to_integer_if_answer_is_valid, check_if_answer_is_among_valid_answers
from config import llama3_8b



class FullLLMChain(dspy.Module):

    #Set up the components of the LLM chain
    def __init__(self):
        self.role_summarizer = dspy.Predict(RoleSummarizer)
        self.relevance_assessor = dspy.Predict(RelevanceAssessor)
        self.expectation_assessor = dspy.ChainOfThought(ExpectationAssessor)

    #Define the flow of data
    def forward(self, excerpt, country_keyword):

        with dspy.context(lm=llama3_8b):
            country_role = self.role_summarizer(
                excerpt = excerpt, 
                country_keyword = 
                country_keyword)
        
        
        relevance_assessment = self.relevance_assessor(
            country_keyword = country_keyword, 
            country_role = country_role.answer
        )

        if relevance_assessment.answer == 'no':
            return relevance_assessment

        expectation_assessment  = self.expectation_assessor(
            country_keyword = country_keyword,
            country_role = country_role.answer,
            excerpt = excerpt)
        
        
        return expectation_assessment