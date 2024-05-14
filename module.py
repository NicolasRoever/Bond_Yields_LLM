import dspy

from signatures.summarizer import RoleSummarizer
from signatures.relevance import RelevanceAssessor
from signatures.sentiment import SentimentAssessor

# Definition of pipeline that produces sentinment score and reasoning, given a text excerpt and a country keyword as input
class SentimentEvaluator(dspy.Module):
    def __init__(self):
        super().__init__()
        self.role_summarizer = dspy.ChainOfThought(RoleSummarizer)
        self.relevance_assessor = dspy.ChainOfThought(RelevanceAssessor)
        self.sentiment_assessor = dspy.ChainOfThought(SentimentAssessor)

    def forward(self, excerpt, country_keyword):
        country_role = self.role_summarizer(country_keyword=country_keyword, excerpt=excerpt).country_role
        relevance = self.relevance_assessor(country_keyword=country_keyword, country_role=country_role).relevance
        if relevance == "yes":
            sentiment_score = self.sentiment_assessor(country_keyword=country_keyword, country_role=country_role).sentiment_score
            sentiment_reasoning = self.sentiment_assessor(country_keyword=country_keyword, country_role=country_role).sentiment_reasoning
        else:
            sentiment_score = "N/A"
            sentiment_reasoning = "N/A"
        return sentiment_score, sentiment_reasoning

# Test pipeline
country_keyword = "greek"
excerpt = "uld be doing as bankers. knowing that we are in a very geared business and we understand that very clearly, and that's why we are so conservative nature. so what i can really tell you sir, is that we are very conservative, we are looking at all the opportunities that we have to provide and to create collateral for ecb, all values, other measures that are available. and of course the fact that the greek banking system is a eurozone country and it is a very solid and has a very high solvency ratio, are very good reasons of why the regulator as well, will take all appropriate measures to safeguard a particular system, given the fact that it has also a very, very low loan-to-deposit ratio overall. so that's why we feel that there is not going to be any kind of crisis that cannot be resolved. a"

test = SentimentEvaluator()(excerpt=excerpt, country_keyword=country_keyword)
print(test)