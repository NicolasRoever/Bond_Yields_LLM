import pandas as pd
import dspy

# Load all snippets
df = pd.read_pickle("df_gpt_sentiment_training_dataset_cleaned.pkl") # Complete dataset
print(df.head())

# Load labeled snippets
df_label = pd.read_excel("300_snippets_transcripts_all_labeled_v002.xlsx", sheet_name="Final Result")
print(df_label.head())

# Split labeled snippets into training and validation sets
# ...

# Construct examples for training and fine-tuning
test_example = dspy.Example(excerpt="a", country_keyword="b", sentiment_score="c", sentiment_reasoning="d").with_inputs("excerpt", "country_keyword")
test_input = test_example.inputs()
test_label = test_example.labels()
print("Example inputs:", test_input)
print("Example labels:", test_label)