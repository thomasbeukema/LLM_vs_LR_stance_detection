from collections import Counter

import pickle
import pandas as pd
import numpy as np

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Load the dataset
df = pd.read_csv('most_differing_entries.tsv', sep='\t')

# List of worker columns
worker_columns = [f'worker_{i}' for i in range(8)]

# Combine the worker annotations into a list of labels for each sentence
df['worker_labels'] = df[worker_columns].values.tolist()
df['majority_label'] = df['worker_labels'].apply(lambda labels: Counter(labels).most_common(1)[0][0])

# Use 'sentence' as input and 'soft_labels' as the target
X = df['sentence']
y = df['majority_label']

# Convert the text data into TF-IDF features
X_tfidf = vectorizer.transform(X)

y_pred = model.predict(X_tfidf)

accuracy = accuracy_score(y, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# Print a classification report for more detailed metrics
print("Classification Report:")
print(classification_report(y, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y, y_pred, labels=np.unique(y)))
print(np.unique(y))