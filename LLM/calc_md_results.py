import pandas as pd

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

df = pd.read_csv('results_md.csv')

# Only extract data from the first prompt
df0 = df[df['prompt'] == 0]
actual = df0['majority_label']
predicted = df0['output']

a0 = accuracy_score(actual, predicted)
print(f'Prompt 1:\nAccuracy: {a0:.2f}')
r0 = classification_report(actual, predicted)
print(r0)

print(confusion_matrix(actual, predicted))

df1 = df[df['prompt'] == 1]
actual = df1['majority_label']
predicted = df1['output']

a1 = accuracy_score(actual, predicted)
print(f'Prompt 2:\nAccuracy: {a1:.2f}')
r1 = classification_report(actual, predicted)
print(r1)

print(confusion_matrix(actual, predicted))

df2 = df[df['prompt'] == 2]
actual = df2['majority_label']
predicted = df2['output']

a2 = accuracy_score(actual, predicted)
print(f'Prompt 3:\nAccuracy: {a2:.2f}')
r2 = classification_report(actual, predicted)
print(r2)

print(confusion_matrix(actual, predicted))
