import fasttext
from tqdm import tqdm

# Paths for birth year
birth_year_input_file = r"C:\Users\Kaanncc\Desktop\birth_year_fasttext.txt"
birth_year_model_path = r"C:\Users\Kaanncc\Desktop\birth_year_model.bin"
birth_year_output_predictions_file = r"C:\Users\Kaanncc\Desktop\birth_year_predictions.txt"

# Paths for political leaning
political_leaning_input_file = r"C:\Users\Kaanncc\Desktop\political_leaning_fasttext.txt"
political_leaning_model_path = r"C:\Users\Kaanncc\Desktop\political_leaning_model.bin"
political_leaning_output_predictions_file = r"C:\Users\Kaanncc\Desktop\political_leaning_predictions.txt"


def generate_predictions(input_file, model_path, output_predictions_file):
    """
    Generates predictions for the input data using a trained FastText model.
    :param input_file: Path to the file containing input data (with ground truth labels).
    :param model_path: Path to the trained FastText model.
    :param output_predictions_file: Path to save the predictions.
    """
    print(f"Loading model from {model_path}...")
    model = fasttext.load_model(model_path)

    print(f"Reading input data from {input_file} and generating predictions...")
    predictions = []

    with open(input_file, "r", encoding="utf-8") as infile:
        lines = infile.readlines()

        for line in tqdm(lines, desc="Processing lines"):
            # Extract text part (ignoring the ground truth label for predictions)
            text = " ".join(line.split()[1:])  # Ignore the first word (label)

            # Predict label and probability
            label, probability = model.predict(text)

            # Store the predicted label and probability
            predictions.append(f"{label[0]}\t{probability[0]:.4f}\n")

    print(f"Saving predictions to {output_predictions_file}...")
    with open(output_predictions_file, "w", encoding="utf-8") as outfile:
        outfile.writelines(predictions)

    print(f"Predictions saved to {output_predictions_file}")


if __name__ == "__main__":
    # Generate predictions for birth year
    generate_predictions(
        birth_year_input_file,
        birth_year_model_path,
        birth_year_output_predictions_file
    )

    # Generate predictions for political leaning
    generate_predictions(
        political_leaning_input_file,
        political_leaning_model_path,
        political_leaning_output_predictions_file
    )
