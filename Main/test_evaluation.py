import numpy as np
from sklearn import metrics
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
def evaluate_classification(targets, predictions):
    r2 = metrics.r2_score(targets, predictions)
    targets = np.round(targets*10).astype(int)
    predictions = predictions * 10
    predictions[predictions < 0] = 0
    predictions[predictions > 10] = 10
    predictions = np.round(predictions).astype(int)
    accuracy = metrics.accuracy_score(targets, predictions)
    # Calculate accuracy
    accuracy = metrics.accuracy_score(targets, predictions) * 100

    # Calculate confusion matrix
    conf_matrix = confusion_matrix(targets, predictions)

    precision, recall, f1, _ = precision_recall_fscore_support(
        targets, predictions, average='weighted', zero_division=0
    )

    return r2, accuracy, conf_matrix, precision, recall, f1

targets = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
predictions = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
train_r2, train_accuracy, train_conf_matrix, train_precision, train_recall, train_f1 = evaluate_classification(targets, predictions)

print(f" Training R2: {train_r2:.4f}, Training Accuracy: {train_accuracy:.2f}%")
print(f"Training Confusion Matrix:\n{train_conf_matrix}")
print(f"Training Precision: {train_precision:.4f}, Recall: {train_recall:.4f}, F1-Score: {train_f1:.4f}")



# Here you can uncomment to explore the csv from the files
# Iterate over each DataFrame in the dictionary
'''for file, df in train_dataframes.items():
    print(f"Data from file: {file}")
    print("Shape:", df.shape)  # Print the shape of the DataFrame
    print("Columns:", df.columns.tolist())  # Print the column names
    print("Info:")
    print(df.info())  # Print basic information about the DataFrame
    print("Head:")
    print(df.head())  # Display the first few rows of the DataFrame
    print("Summary statistics:")
    print(df.describe())  # Display summary statistics for numerical columns
    print("Unique values per column:")
    print(df.nunique())  # Display the number of unique values in each column
    print("\n")'''
