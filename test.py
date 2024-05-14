import dspy
from signatures.summarizer import RoleSummarizer

test_model = dspy.Predict(RoleSummarizer)

sentence = "Germany's economy is expected to grow by 3% this year."

test = test_model(country_keyword="Germany", excerpt=sentence).country_role
print(test)