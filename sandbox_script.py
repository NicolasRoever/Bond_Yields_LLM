import groq
import os
from dotenv import load_dotenv
import dspy
import numpy as np
from dspy.evaluate.metrics import answer_exact_match
import pandas as pd
from IPython.core.display import Markdown
from dspy import Example
from dspy.teleprompt import BootstrapFewShot
from dspy.teleprompt import BootstrapFewShotWithRandomSearch
from dspy.evaluate import Evaluate
from module_v002 import FullLLMChain
from optimize import passage_similarity_metric, custom_evaluation_function, similar_score_metric, evaluate_expectations_metric
import pdb

llama3_8b = dspy.OllamaLocal(model = "llama3:8b",
                             temperature = 0,
                             max_tokens = 800)
dspy.settings.configure(lm=llama3_8b)

    
full_llm_chain = FullLLMChain()

data = pd.read_excel("/Users/nicolasroever/Dropbox/5_Bond yield spreads/project_python/bond_yields_llm/data/300_snippets_transcripts_all_labeled_v002.xlsx")
snippet_list = data["Snippet"].tolist()
keyword_list = data["Keyword"].tolist()
data["Relevance_Score_Yes_No"] = np.where(data["Final Relevance Score"] == 1, 'no', 'yes')

examples = [dspy.Example(excerpt=row["Snippet"], country_keyword = row["Keyword"], answer=row["Final Combined"]).with_inputs("excerpt","country_keyword") for idx, row in data.iloc[0:300, ].iterrows()]
trainset = examples[0:9]
valset = examples[57:59]


config = dict(max_bootstrapped_demos=4, max_labeled_demos=2, num_candidate_programs = 5)
teleprompter = BootstrapFewShotWithRandomSearch(metric=evaluate_expectations_metric, **config)
optimized_llm = teleprompter.compile(full_llm_chain, trainset=trainset, valset=valset)