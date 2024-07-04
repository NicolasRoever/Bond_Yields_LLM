
import dspy
from signatures.summarizer import RoleSummarizer
from signatures.relevance import RelevanceAssessor
from signatures.expectations import ExpectationAssessor
from config import convert_to_integer_if_answer_is_valid, check_if_answer_is_among_valid_answers
from config import llama3_8b
import pandas as pd
import random



class FullLLMChain(dspy.Module):
    # Set up the components of the LLM chain
    def __init__(self):
        super().__init__()
        self.role_summarizer = dspy.Predict(RoleSummarizer)
        self.relevance_assessor = dspy.Predict(RelevanceAssessor)
        self.expectation_assessor = dspy.ChainOfThought(ExpectationAssessor)
    # Define the flow of data
    def forward(self, excerpt, country_keyword):
        outputs = {}
        random_number = random.random()

 
        # Role summarizer
        country_role = self.role_summarizer(
            excerpt=excerpt,
            country_keyword=country_keyword, 
            config=dict(temperature=0.7 + 0.0001 * random_number)
        )


        outputs["answer_role_summarizer"] = country_role.answer

        # Relevance assessor
        relevance_assessment = self.relevance_assessor(
            country_keyword=country_keyword,
            country_role=country_role.answer, 
            config=dict(temperature=0.7 + 0.0001 * random_number)
        )
  
        outputs["answer_relevance_assessor"] = relevance_assessment.answer

        if relevance_assessment.answer == 'no':

            outputs["answer_expectation_assessor"] = pd.NA
            outputs["rationale_expectation_assessor"] = pd.NA
            return outputs

        # Expectation assessor
        expectation_assessment = self.expectation_assessor(
            country_keyword=country_keyword,
            country_role=country_role.answer,
            excerpt=excerpt, 
            config=dict(temperature=0.7 + 0.0001 * random_number)
        )
 
        outputs["answer_expectation_assessor"] = expectation_assessment.answer
        outputs["rationale_expectation_assessor"] = expectation_assessment.rationale


        return outputs