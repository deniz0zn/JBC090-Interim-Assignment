from sklearn.metrics import jaccard_score, classification_report, accuracy_score
from tqdm import tqdm

# Paths to original and prediction files
birth_year_original_file = r"C:\Users\Kaanncc\Desktop\birth_year_fasttext.txt"
birth_year_predictions_file = r"C:\Users\Kaanncc\Desktop\birth_year_predictions.txt"
political_leaning_original_file = r"C:\Users\Kaanncc\Desktop\political_leaning_fasttext.txt"
political_leaning_predictions_file = r"C:\Users\Kaanncc\Desktop\political_leaning_predictions.txt"

def evaluate_predictions(original_file, predictions_file, label=""):
    """
    Evaluates predictions using Jaccard Coefficient and classification metrics.
    :param original_file: Path to the file with original data and ground truth labels.
    :param predictions_file: Path to the file with predicted labels.
    :param label: The label for the dataset (e.g., "Birth Year", "Political Leaning").
    """
    # Read ground truth and predictions
    with open(original_file, "r", encoding="utf-8", errors="ignore") as orig, \
         open(predictions_file, "r", encoding="utf-8", errors="ignore") as pred:
        ground_truth_labels = []
        predicted_labels = []

        print(f"Reading and comparing data for {label}...")
        total_lines = sum(1 for _ in open(original_file, encoding="utf-8", errors="ignore"))
        for orig_line, pred_line in tqdm(zip(orig, pred), desc=f"Processing {label} lines", total=total_lines):
            # Extract ground truth label
            ground_truth_label = orig_line.split()[0]  # First part is the label
            ground_truth_labels.append(ground_truth_label)

            # Extract predicted label
            predicted_label = pred_line.split("\t")[0]  # First part is the predicted label
            predicted_labels.append(predicted_label)

    # Jaccard Score (micro-averaged for multi-class)
    print(f"\nCalculating Jaccard Coefficient for {label}...")
    jaccard = jaccard_score(
        ground_truth_labels,
        predicted_labels,
        average="micro"
    )
    print(f"Jaccard Coefficient (micro) for {label}: {jaccard:.4f}")

    # Classification Metrics
    print(f"\nClassification Metrics for {label}:")
    print(classification_report(ground_truth_labels, predicted_labels))

    # Overall Accuracy
    accuracy = accuracy_score(ground_truth_labels, predicted_labels)
    print(f"Accuracy for {label}: {accuracy:.4f}")


if __name__ == "__main__":
    # Evaluate predictions for birth year
    evaluate_predictions(birth_year_original_file, birth_year_predictions_file, label="Birth Year")

    # Evaluate predictions for political leaning
    evaluate_predictions(political_leaning_original_file, political_leaning_predictions_file, label="Political Leaning")
