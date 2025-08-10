import csv
import sys

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py data")

    # Load data from spreadsheet and split into train and test sets
    evidence, labels = load_data(sys.argv[1])
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )

    # Train model and make predictions
    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)
    sensitivity, specificity = evaluate(y_test, predictions)

    # Print results
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")


def load_data(filename):
    evidence = []
    labels = []
    for line in csv.reader(open(filename, "r")): 
        if line[0] == "Administrative":
            continue  # Skip header row
        labels.append(1 if line[17].strip().upper() == "TRUE" else 0) #label
        #loop over other data and add it to the evidence list as int or float
        evidence.append([
            int(line[0]), # Administrative
            float(line[1]), # Administrative_Duration
            int(line[2]), # Informational
            float(line[3]), # Informational_Duration
            int(line[4]), # ProductRelated
            float(line[5]), # ProductRelated_Duration
            float(line[6]), # BounceRates
            float(line[7]), # ExitRates
            float(line[8]), # PageValues
            float(line[9]), # SpecialDay
            ["Jan", "Feb", "Mar", "Apr", "May", "June", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"].index(line[10]), # Month
            int(line[11]), # OperatingSystems
            int(line[12]), # Browser
            int(line[13]), # Region
            int(line[14]), # TrafficType
            1 if line[15] == "Returning_Visitor" else 0, # VisitorType
            1 if line[16] == "TRUE" else 0 # Weekend
        ])
    return (evidence, labels)

def train_model(evidence, labels):
    model = KNeighborsClassifier(n_neighbors=1)
    model.fit(evidence, labels)
    return model


def evaluate(labels, predictions):
    sensitivity = 0.0
    specificity = 0.0

    true_positives = sum(1 for i in range(len(labels)) if labels[i] == 1 and predictions[i] == 1)
    true_negatives = sum(1 for i in range(len(labels)) if labels[i] == 0 and predictions[i] == 0)
    false_positives = sum(1 for i in range(len(labels)) if labels[i] == 0 and predictions[i] == 1)
    false_negatives = sum(1 for i in range(len(labels)) if labels[i] == 1 and predictions[i] == 0)

    if true_positives + false_negatives > 0:
        sensitivity = true_positives / (true_positives + false_negatives)
    if true_negatives + false_positives > 0:
        specificity = true_negatives / (true_negatives + false_positives)
    return (sensitivity, specificity)


if __name__ == "__main__":
    main()
