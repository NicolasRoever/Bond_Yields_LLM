import pandas as pd
from module import sentiment_evaluator
from dspy.teleprompt import BootstrapFewShot
from data.preprocess import train_set, val_set
from dspy.evaluate import Evaluate

# Define metric
def metric(truth, pred, trace=None):
    truth_score = truth.sentiment_score
    pred_score = pred.sentiment_score

    if truth_score == pred_score:
        return True
    else:
        return False
    
config = dict(max_bootstrapped_demos=4, max_labeled_demos=8, max_errors=1, max_rounds=1)

optimizer = BootstrapFewShot(metric=metric, **config)

optimized_SentimentEvaluator = optimizer.compile(student=sentiment_evaluator, trainset=train_set, valset=val_set)

evaluate = Evaluate(metric=metric, devset=train_set, num_threads=1, display_progress=True, display_table=5)
evaluate(optimized_SentimentEvaluator)

optimized_SentimentEvaluator.save(path="/tmp/model_gpt4o.json")