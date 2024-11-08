from collections import Counter

import openai
import pandas as pd

client = openai.Client(
    api_key=""
)

# Load the dataset
df = pd.read_csv('md.tsv', sep='\t')

# List of worker columns
worker_columns = [f'worker_{i}' for i in range(8)]

# Combine the worker annotations into a list of labels for each sentence
df['worker_labels'] = df[worker_columns].values.tolist()
df['majority_label'] = df['worker_labels'].apply(lambda labels: Counter(labels).most_common(1)[0][0])

# Use 'sentence' as input and 'soft_labels' as the target
X_md = df['sentence']
y_md = df['majority_label']

prompts = [
    "You're tasked with classifying pieces of text. You have to label a piece of text 'agrees', 'neutral' or 'disagrees' whether the text indicates that the author agrees, is neutral towards or disagrees with the sentence 'Global warming is a threat.'. Only answer with the label, and no other text. The sentence: ",
    "Classify the stance of the following text towards the statement: ‘Global warming is a threat.’. Your classification must be one of three labels: ‘agrees’, ‘neutral’, or ‘disagrees’. Only respond with one of the labels, and no other text. The sentence: ",
    "Classify the stance of the given text as ‘agrees’, ‘neutral’, or ‘disagrees’ in relation to the statement: ‘Global warming is a threat.’. Return only the appropriate label, no explanation. The sentence:"
]

results = []

for prompt in prompts:
    for X, y in pd.concat([X_md, y_md], axis=1).values:
        completion = client.chat.completions.create(
            model="gpt-4o",
            temperature=0,
            messages=[
                {"role": "user", "content": f"{prompt}{X}"},
            ]
        )

        results.append({
            "sentence": X,
            "majority_label": y,
            "prompt": prompts.index(prompt),
            "output": completion.choices[0].message.content.strip()
        })
        print(results[-1])

# Write results to a CSV file
results_df = pd.DataFrame(results)
results_df.to_csv('results_md.csv', index=False)