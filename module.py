import dspy
import pandas as pd

from signatures.summarizer import RoleSummarizer
from signatures.relevance import RelevanceAssessor
from signatures.sentiment import SentimentAssessor
from data.preprocess import examples

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
        relevance_rationale = self.relevance_assessor(country_keyword=country_keyword, country_role=country_role).rationale
        sentiment_rationale = ""
        if relevance == "yes":
            sentiment_score = self.sentiment_assessor(country_keyword=country_keyword, country_role=country_role).sentiment_score
            sentiment_rationale = self.sentiment_assessor(country_keyword=country_keyword, country_role=country_role).rationale
        else:
            sentiment_score = 99
        return {'country_role': country_role, 'relevance': relevance, 'relevance_rationale': relevance_rationale, 'sentiment_score': sentiment_score, 'sentiment_rationale': sentiment_rationale}
        
sentiment_evaluator = SentimentEvaluator()

results = []

test = examples[:1]

for item in test:
    sentiment_result = sentiment_evaluator(excerpt=item['excerpt'], country_keyword=item['country_keyword'])
    sentiment_result['sentiment_rationale'] = sentiment_result['sentiment_rationale'].replace('\n', ' ')
    sentiment_result['relevance_rationale'] = sentiment_result['relevance_rationale'].replace('\n', ' ')
    sentiment_result['country_role'] = sentiment_result['country_role'].replace('\n', ' ')
    results.append({
        'Snippet_ID': item['snippet_id'],
        'Snippet': item['excerpt'],
        'Keyword': item['country_keyword'],
        **sentiment_result
    })

df = pd.DataFrame(results)
df.to_csv('output/sentiment_results_llm.csv', index=False)