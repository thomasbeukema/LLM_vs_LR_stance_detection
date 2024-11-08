from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

import pickle

df = pd.read_csv('other_entries.tsv', sep='\t')

# List of worker columns
worker_columns = [f'worker_{i}' for i in range(8)]

# Combine the worker annotations into a list of labels for each sentence
df['worker_labels'] = df[worker_columns].values.tolist()
df['majority_label'] = df['worker_labels'].apply(lambda labels: Counter(labels).most_common(1)[0][0])

# Use 'sentence' as input and 'soft_labels' as the target
X = df['sentence']
y = df['majority_label']

# Split the data into training and test sets (80% training, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
df_test = pd.concat([X_test, y_test], axis=1)
df_test.to_csv('test.csv', index=False)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=1)
df_val = pd.concat([X_val, y_val], axis=1)
df_val.to_csv('val.csv', index=False)

# Convert the text data into TF-IDF features
vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words='english', max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)
X_val_tfidf = vectorizer.transform(X_val)

param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'penalty': ['l2'],  # lbfgs only supports l2 or none
    'solver': ['lbfgs', 'newton-cg', 'saga'],
    'max_iter': [1000, 1500, 2000]  # Increase iterations if needed
}

# Initialize the Logistic Regression model with multinomial strategy
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)

grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')

# Fit the grid search to the training data
grid_search.fit(X_train_tfidf, y_train)

# Get the best model hyperparameters
best_model = grid_search.best_estimator_
print("Best hyperparameters: ", grid_search.best_params_)

# Make predictions on the test set
y_pred = best_model.predict(X_val_tfidf)

# Evaluate the model
accuracy = accuracy_score(y_val, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# Print a classification report for more detailed metrics
print("Classification Report:")
print(classification_report(y_val, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_val, y_pred))
print(np.unique(y_val))

with open('model.pkl', 'wb') as model_file:
    pickle.dump(best_model, model_file)

with open('vectorizer.pkl', 'wb') as vectorizer_file:
    pickle.dump(vectorizer, vectorizer_file)