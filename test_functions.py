import pytest
import re
import dspy
from config import convert_to_integer_if_answer_is_valid
from optimize import extract_last_integer_from_string, custom_evaluation_metric
from signatures.expectations import ExpectationAssessor
from signatures.summarizer import RoleSummarizer
from signatures.relevance import RelevanceAssessor, extract_relevance_as_yes_or_no
from module_v002 import FullLLMChain


llama3_8b = dspy.OllamaLocal(model = "llama3:8b",
                             temperature = 0,
                             max_tokens = 800)

dspy.settings.configure(lm=llama3_8b)

def test_extract_last_integer_from_string_with_string():

    example_input = "Country Keyword: France\n\nCountry Role: Country mentioned as a priority for European expansion\n\nExcerpt: ...we still view European expansion into countries such as France and Germany as a high priority for Concordia.\n\nReasoning: Let's think step by step in order to. The company views European expansion, including France, as a high priority, indicating that France plays a significant role in their growth strategy. This suggests that the country is considered important for the company's future plans and growth prospects. Therefore, I assess the expectation towards the solvency of France as: 1"

    expected_result = 1
    actual_result = extract_last_integer_from_string(example_input)

    assert expected_result == actual_result


def test_extract_last_integer_from_string_with_int():
    example_input = "-2"
    expected_result = -2
    actual_result = extract_last_integer_from_string(example_input)

    assert expected_result == actual_result


def test_custom_evaluation_metric():
    # Test when prediction and actual are the same
    assert custom_evaluation_metric(3, 3) == 1.0
    assert custom_evaluation_metric(-2, -2) == 1.0

    # Test when prediction and actual are in the same direction
    assert custom_evaluation_metric(2, 3) == 0.5
    assert custom_evaluation_metric(-2, -3) == 0.5

    # Test when prediction and actual are not in the same direction
    assert custom_evaluation_metric(2, -3) == 0.0
    assert custom_evaluation_metric(-2, 3) == 0.0

    # Test when prediction or actual is zero
    assert custom_evaluation_metric(0, 3) == 0.0
    assert custom_evaluation_metric(3, 0) == 0.0
    assert custom_evaluation_metric(0, 0) == 1.0

def test_convert_to_integer_if_answer_is_valid():
    assert convert_to_integer_if_answer_is_valid("-2") == -2
    assert convert_to_integer_if_answer_is_valid("-1") == -1
    assert convert_to_integer_if_answer_is_valid("0") == 0
    assert convert_to_integer_if_answer_is_valid("1") == 1
    assert convert_to_integer_if_answer_is_valid("2") == 2
    assert convert_to_integer_if_answer_is_valid("99") == 99
    assert convert_to_integer_if_answer_is_valid("Hello") == "Hello"
    assert convert_to_integer_if_answer_is_valid("10") == "10"  # Edge case not in the specified list


#--- Test the Individual Signatures ---#

class RoleSummarizerTestingPurpose(dspy.Module):

    def __init__(self):
        self.role_summarizer = dspy.Predict(RoleSummarizer)

    def forward(self, excerpt, country_keyword):
        country_role = self.role_summarizer(excerpt=excerpt, country_keyword=country_keyword)
        return country_role


def test_role_summarizer():

    role_summarizer = RoleSummarizerTestingPurpose()

    text_excerpt = "France has shown significant GDP growth in the last quarter. We expect this trend to continue."

    test_country_keyword = "France"

    expected_result = 'France: The country is mentioned as having shown significant GDP growth, indicating its economic performance and potential for future growth.'

    with dspy.context(lm=llama3_8b):
        actual_result = role_summarizer(excerpt=text_excerpt, country_keyword=test_country_keyword)

    assert expected_result == actual_result.answer


class RelevanceAssessorTestingPurpose(dspy.Module):

    def __init__(self):
        self.relevance_assessor = dspy.Predict(RelevanceAssessor)

    def forward(self, country_keyword, country_role):
        relevance = self.relevance_assessor(country_keyword=country_keyword, country_role=country_role)
        return relevance

def test_relevance_assessor():

    relevance_assessor = RelevanceAssessorTestingPurpose()

    test_country_keyword = "France"
    test_country_role = "France: The country is mentioned as having shown significant GDP growth, indicating its economic performance and potential for future growth."

    expected_result = 'yes'

    with dspy.context(lm=llama3_8b):
        actual_result = relevance_assessor(country_keyword=test_country_keyword, country_role=test_country_role)

        relevance_yes_no = extract_relevance_as_yes_or_no(actual_result.answer)

    assert expected_result == relevance_yes_no


class ExpectationAssessorTestingPurpose(dspy.Module):

    def __init__(self):
        self.expectation_assessor = dspy.ChainOfThought(ExpectationAssessor)

    def forward(self, country_keyword, country_role, excerpt):

        expectation_assessment = self.expectation_assessor(country_keyword=country_keyword, country_role=country_role, excerpt=excerpt)

        return expectation_assessment
    

def test_expectation_assessor():

    expectation_assessor = ExpectationAssessorTestingPurpose()

    test_country_keyword = "France"
    test_country_role = "France: The country is mentioned as having shown significant GDP growth, indicating its economic performance and potential for future growth."
    test_excerpt = "France has shown significant GDP growth in the last quarter. We expect this trend to continue."

    expected_answer = 1

    expected_rationale = '1' # This is because lama is a shitty model :)

    with dspy.context(lm=llama3_8b):
        actual_result = expectation_assessor(country_keyword=test_country_keyword, country_role=test_country_role, excerpt=test_excerpt)

    assert expected_answer == convert_to_integer_if_answer_is_valid(actual_result.answer)

    assert expected_rationale == actual_result.rationale


 #--- Test the Full LLM Chain ---#

def test_full_llm_chain_standard_example_1():

    full_llm_chain = FullLLMChain()

    test_excerpt = "France has shown significant GDP growth in the last quarter. We expect this trend to continue."

    test_country_keyword = "France"

    with dspy.context(lm=llama3_8b):
        actual_result = full_llm_chain(excerpt=test_excerpt, country_keyword=test_country_keyword)

    assert convert_to_integer_if_answer_is_valid(actual_result.answer) == 1
    assert actual_result.rationale == '1' # This is because lama is a shitty model :)


def test_full_llm_chain_standard_example_2():

    full_llm_chain = FullLLMChain()

    # Second test case
    test_excerpt = "Germany's economy has been struggling recently, with GDP growth slowing down."
    test_country_keyword = "Germany"

    with dspy.context(lm=llama3_8b):
        actual_result = full_llm_chain(excerpt=test_excerpt, country_keyword=test_country_keyword)

    breakpoint()
    assert actual_result.answer == -1
    assert actual_result.rationale == '-1' # This is because lama is a shitty model :)


class MockClassNotRelevant:
    def __init__(self, country_keyword, country_role):
        self.country_keyword = country_keyword
        self.country_role = country_role
        self.answer = "no"
        self.successfull_test = "Yes"


class FullLLMChainNotRelevantTest(dspy.Module):

    def __init__(self):
        self.role_summarizer = dspy.Predict(RoleSummarizer)
        self.relevance_assessor = MockClassNotRelevant
        self.expectation_assessor = dspy.ChainOfThought(ExpectationAssessor)

    def forward(self, excerpt, country_keyword):

        country_role = self.role_summarizer(excerpt=excerpt, country_keyword=country_keyword)

        relevance_assessment = self.relevance_assessor(country_keyword=country_keyword, country_role=country_role.answer)

        if relevance_assessment.answer == 'no':
            return relevance_assessment


def test_if_snippet_is_not_relevant():

    full_llm_chain = FullLLMChainNotRelevantTest()

    test_excerpt = "Germany's economy has been struggling recently, with GDP growth slowing down."

    test_country_keyword = "Germany"

    with dspy.context(lm=llama3_8b):
        actual_result = full_llm_chain(excerpt=test_excerpt, country_keyword=test_country_keyword)

    assert actual_result.successfull_test == "Yes"

    


    
    




