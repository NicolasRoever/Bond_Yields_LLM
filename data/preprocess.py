import pandas as pd
import dspy

# Load all snippets
df = pd.read_pickle("data/df_gpt_sentiment_training_dataset_cleaned.pkl") # Complete dataset

# Load labeled snippets
df_label = pd.read_excel("data/300_snippets_transcripts_all_labeled_v002.xlsx", sheet_name="Final Result")
df_label_selected = df_label[["Keyword", "Snippet", "Final Combined"]]

# Split labeled snippets into training and validation sets
df_label_selected = df_label_selected.sample(frac=1, random_state=42)

examples = []
for index, row in df_label_selected.iterrows():
    examples.append(dspy.Example(excerpt=row['Snippet'], 
                                 country_keyword=row['Keyword'], 
                                 sentiment_score=row['Final Combined']).with_inputs('excerpt', 'country_keyword'))

train_set = examples[:200]
val_set = examples[200:250]
test_set = examples[250:300]