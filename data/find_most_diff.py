from collections import Counter
import math

import matplotlib.pyplot as plt
import pandas as pd

number = 200

# Load the dataset
file_path = 'GWSD.tsv'  # Replace with the path to your file
df = pd.read_csv(file_path, sep='\t')

# Select the relevant worker columns
worker_columns = ['worker_0', 'worker_1', 'worker_2', 'worker_3', 'worker_4', 'worker_5', 'worker_6', 'worker_7']

# Function to calculate the number of unique labels (disagreement score)
def calculate_entropy(row):
    unique_labels = row[worker_columns]
    counts = Counter(unique_labels)

    total_counts = len(counts)

    probabilities = [count / total_counts for count in counts.values()]

    entropy = -sum(p * math.log2(p) for p in probabilities if p > 0)

    return entropy

# Apply the disagreement function to each row
df['disagreement_score'] = df.apply(calculate_entropy, axis=1)

# Sort the dataframe by disagreement score in descending order and take the top
most_differing_entries = df.sort_values(by='disagreement_score', ascending=False).head(number)

# Calculate avg entropy and print
avg_entropy = df['disagreement_score'].mean()
std_dev = df['disagreement_score'].std()
lowest = df['disagreement_score'].min()
highest = df['disagreement_score'].max()

# Save a plot of the distribution of disagreement scores
df['disagreement_score'].hist(bins=200)
plt.xlabel('Entropy')
plt.ylabel('Frequency')
plt.title('Distribution of entropy')
plt.savefig('disagreement_scores.png')

plt.close()

# Save a plot of the distribution of disagreement scores for the most differing entries
most_differing_entries['disagreement_score'].hist(bins=20)
plt.xlabel('Disagreement Score')
plt.ylabel('Frequency')
plt.title('Distribution of Disagreement Scores for Most Differing Entries')
plt.savefig('most_differing_disagreement_scores.png')

# Calculate same values for most differing entries
avg_entropy_most_differing = most_differing_entries['disagreement_score'].mean()
std_dev_most_differing = most_differing_entries['disagreement_score'].std()
lowest_most_differing = most_differing_entries['disagreement_score'].min()
highest_most_differing = most_differing_entries['disagreement_score'].max()

# Calculate same values for other entries
other_entries = df.drop(most_differing_entries.index)
avg_entropy_other = other_entries['disagreement_score'].mean()
std_dev_other = other_entries['disagreement_score'].std()
lowest_other = other_entries['disagreement_score'].min()
highest_other = other_entries['disagreement_score'].max()

print(f"Average disagreement score: {avg_entropy:.2f}")
print(f"Standard deviation of disagreement scores: {std_dev:.2f}")
print(f"Lowest disagreement score: {lowest:.2f}")
print(f"Highest disagreement score: {highest:.2f}")

print()

print(f"Average disagreement score for most differing entries: {avg_entropy_most_differing:.2f}")
print(f"Standard deviation of disagreement scores for most differing entries: {std_dev_most_differing:.2f}")
print(f"Lowest disagreement score for most differing entries: {lowest_most_differing:.2f}")
print(f"Highest disagreement score for most differing entries: {highest_most_differing:.2f}")

print()
print(f"Average disagreement score for other entries: {avg_entropy_other:.2f}")
print(f"Standard deviation of disagreement scores for other entries: {std_dev_other:.2f}")
print(f"Lowest disagreement score for other entries: {lowest_other:.2f}")
print(f"Highest disagreement score for other entries: {highest_other:.2f}")

# Output the results
output_file = 'most_differing_entries.tsv'
most_differing_entries.to_csv(output_file, sep='\t', index=False)

# Also save the other sentences to another file, without the most differing entries
other_entries = df.drop(most_differing_entries.index)
print(len(other_entries))
other_file = 'other_entries.tsv'
other_entries.to_csv(other_file, sep='\t', index=False)

print(f"Saved the {number} most differing entries to {output_file}")