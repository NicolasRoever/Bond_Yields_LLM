import pandas as pd
from dspy.teleprompt import BootstrapFewShot
from dspy.evaluate import Evaluate
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from IPython.core.display import Markdown
import pdb
import re
import io
import contextlib




def evaluate_expectations_metric(reference_response, generated_response, trace=None):
    """Calculate the similarity between a reference response and a generated response.

    Parameters:
    - reference_response (psdy.Example): The predefined response to compare against - which is the standard dspy.Example object,
    - generated_response (psdy.Example): The generated response.

    Returns:
    - boolen: True if the reference response and the generated response are the same, False otherwise.
    """

    if generated_response["answer_relevance_assessor"] == 'no':

        if reference_response.sentiment_score == 99:
            return 1
        
        if reference_response.sentiment_score == 0:
            return 1
        
        else:
            return 0

    else:
        pred_expectation_score = extract_last_integer_from_string(generated_response["answer_expectation_assessor"])

    return custom_evaluation_metric(pred_expectation_score, int(reference_response.sentiment_score))

    
def custom_evaluation_metric(pred, actual):
    """
    Custom evaluation metric that gives 0.5 points if the prediction is in the right direction
    and 1 point if the prediction is the same as the actual value.

    Parameters:
    - pred (int): The predicted score.
    - actual (int): The actual score.

    Returns:
    - float: The evaluation score.
    """

    if pred == None:
        print("None value in prediction")
        return 0
    
    #If prediction is 0 and actual value is 99, return 1 (this is the no relevance case)
    if pred == 0 and actual == 99:
        return 1
       
    # If the prediction is the same as the actual value, return 1
    if pred == actual:
        return 1

    # If the prediction is in the right direction, return 0.5
    if (pred > 0 and actual > 0) or (pred < 0 and actual < 0):
        return 0.5

    # If the prediction is not in the right direction, return 0
    return 0



def passage_similarity_metric(reference_text, generated_text, trace=None):
    """
    Calculate the similarity between a reference text and a generated text.
    
    Parameters:
    - reference_text (str): The predefined text passage to compare against.
    - generated_text (str): The generated text passage.
    
    Returns:
    - float: Similarity score between the reference text and the generated text.
    """

    # Combine reference text and generated text for vectorization
    generated_country_role = generated_text.answer.split("Answer:", 1)[1].strip()
    example_country_role = reference_text.answer
    texts = [example_country_role, generated_country_role]
    
    # Create TF-IDF vectorizer and fit-transform the texts
    vectorizer = TfidfVectorizer().fit_transform(texts)
    
    # Calculate cosine similarity between the reference text and the generated text
    similarity_score = cosine_similarity(vectorizer[0:1], vectorizer[1:2]).flatten()[0]
    
    return similarity_score

def custom_evaluation_function(validation_set, llm, show_examples = 5, metric_for_evaluation = "passage_similarity"):
    """
    This function prints validation examples in a readable format.

    Args:
    - validation_set (dspy.examples): A list of validation examples in dspy example format
    """
    for example in validation_set[0:show_examples]:

        prediction = llm(excerpt=example.excerpt, country_keyword=example.country_keyword)

        if metric_for_evaluation == "passage_similarity":
            metric = passage_similarity_metric(example, prediction)

        elif metric_for_evaluation == "similar_score":
            metric = similar_score_metric(example, prediction)

        elif metric_for_evaluation == "evaluate_expectations":

            metric = evaluate_expectations_metric(example, prediction)
        
        readable_prediction = prediction.answer

        display(Markdown(f"**Human Coded Answer:** {example.answer}\n\n**LLM Answer:** {readable_prediction}\n\n**Metric:** {metric:.2f}\n\n---\n\n"))



def extract_last_integer_from_string(text):
    """
    Extract the last integer from a string.

    Parameters:
    - text (str): The input string.

    Returns:
    - int: The last integer in the string, or None if no integer is found.
    """

    # Find all integers in the string
    integers = re.findall(r'-?\d+', text)

    # If no integer is found, return None
    if not integers:
        return None

    # Return the last integer as an integer
    return int(integers[-1])

def create_dataframe_with_validation_results(validation_examples, llm_chain):
    """
    Evaluates the validation examples using the provided LLM chain function and returns a DataFrame with the results.

    Parameters:
    validation_examples (list): A list of validation examples.
    llm_chain_function (function): The LLM chain function used to generate predictions.

    Returns:
    pd.DataFrame: A DataFrame containing the Snippet_ID, Prediction, Reference_Response, and Score for each example.
    """
    results = []

    # Process the validation examples
    for i,x in enumerate(validation_examples):
        pred = llm_chain(excerpt=x.excerpt, country_keyword=x.country_keyword)
        score = evaluate_expectations_metric(reference_response=x, generated_response=pred)
        
        # Append the results to the list
        results.append({
            'Snippet_ID': x.snippet_id,
            'Excerpt': x.excerpt,
            'Answer_Role_Summarizer': pred["answer_role_summarizer"],
            'Answer_Relevance_Assessor': pred["answer_relevance_assessor"],
            'Prediction': pred["answer_expectation_assessor"],
            'Rationale_for_Prediction': pred["rationale_expectation_assessor"],
            'Reference_Response': x.sentiment_score,
            'Evaluation_Score': score
        })

        print(f"Finished validation example {i} out of {len(validation_examples) } ({i/len(validation_examples)*100:.2f}%)")


    return pd.DataFrame(results)

def similar_score_metric(reference_response, generated_response, trace=None):
    """
    Calculate the similarity between a reference response and a generated response.

    Parameters:
    - reference_response (psdy.Example): The predefined response to compare against, 
    - generated_response (psdy.Example): The generated response.

    Returns:
    - boolen: True if the reference response and the generated response are the same, False otherwise.
    """

    truth_score = reference_response.answer
    pred_score = generated_response.answer.split("Answer:", 1)[1].strip()

    if truth_score == pred_score:
        return True
    else:
        return False
    
def capture_and_save_output(func, file_name, *args, **kwargs):
    """
    Captures the printed output of a function and saves it to a text file.
    
    Parameters:
    func (callable): The function whose output is to be captured.
    file_name (str): The name of the file to save the output.
    *args: Variable length argument list to pass to the function.
    **kwargs: Arbitrary keyword arguments to pass to the function.
    """
    output_buffer = io.StringIO()
    with contextlib.redirect_stdout(output_buffer):
        func(*args, **kwargs)
    
    # Get the output as a string
    captured_output = output_buffer.getvalue()

    # Write the captured output to the file
    with open(file_name, "w") as file:
        file.write(captured_output)


# LATER FOR OPTIMIZATION: optimize input/output examples for prompt of sentiment llm based on hand-coded examples

# # Define metric
# def metric(truth, pred, trace=None):
#     truth_score = truth.sentiment_score
#     pred_score = pred.sentiment_score

#     if truth_score == pred_score:
#         return True
#     else:
#         return False
    
# config = dict(max_bootstrapped_demos=4, max_labeled_demos=8, max_errors=1, max_rounds=1)

# optimizer = BootstrapFewShot(metric=metric, **config)

# optimized_SentimentEvaluator = optimizer.compile(student=sentiment_evaluator, trainset=train_set, valset=val_set)

# evaluate = Evaluate(metric=metric, devset=train_set, num_threads=1, display_progress=True, display_table=5)
# evaluate(optimized_SentimentEvaluator)

# optimized_SentimentEvaluator.save(path="/tmp/model_gpt4o.json")