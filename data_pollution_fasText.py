import os
import random
import fasttext
from tqdm import tqdm

# Paths to FastText-formatted data files
birth_year_input_txt = r"C:\Users\Kaanncc\Desktop\birth_year_fasttext.txt"
political_leaning_input_txt = r"C:\Users\Kaanncc\Desktop\political_leaning_fasttext.txt"

# Output paths for splits
birth_year_train_txt = r"C:\Users\Kaanncc\Desktop\birth_year_train.txt"
birth_year_val_txt = r"C:\Users\Kaanncc\Desktop\birth_year_val.txt"
birth_year_test_txt = r"C:\Users\Kaanncc\Desktop\birth_year_test.txt"

political_leaning_train_txt = r"C:\Users\Kaanncc\Desktop\political_leaning_train.txt"
political_leaning_val_txt = r"C:\Users\Kaanncc\Desktop\political_leaning_val.txt"
political_leaning_test_txt = r"C:\Users\Kaanncc\Desktop\political_leaning_test.txt"


def split_data(input_file, train_file, val_file, test_file, split_ratios=(0.8, 0.1, 0.1)):
    """
    Splits the data into training, validation, and test sets.
    :param input_file: Path to the input data file.
    :param train_file: Path to save the training data.
    :param val_file: Path to save the validation data.
    :param test_file: Path to save the test data.
    :param split_ratios: Tuple of ratios for train, val, test split.
    """
    print(f"Splitting data from {input_file}...")

    # Read the input data
    with open(input_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Shuffle the data
    random.shuffle(lines)

    # Calculate split sizes
    total = len(lines)
    train_size = int(split_ratios[0] * total)
    val_size = int(split_ratios[1] * total)

    # Split the data
    train_data = lines[:train_size]
    val_data = lines[train_size:train_size + val_size]
    test_data = lines[train_size + val_size:]

    # Save the splits
    with open(train_file, "w", encoding="utf-8") as f:
        f.writelines(train_data)
    with open(val_file, "w", encoding="utf-8") as f:
        f.writelines(val_data)
    with open(test_file, "w", encoding="utf-8") as f:
        f.writelines(test_data)

    print(f"Data split completed:\n"
          f" - Training data: {len(train_data)} samples\n"
          f" - Validation data: {len(val_data)} samples\n"
          f" - Test data: {len(test_data)} samples\n")


def train_fasttext_model(train_file, val_file, output_model):
    """
    Trains and evaluates a FastText model.
    :param train_file: Path to training data.
    :param val_file: Path to validation data.
    :param output_model: Path to save the trained model.
    """
    print(f"Training FastText model with data from {train_file}...")

    # Train the model
    model = fasttext.train_supervised(
        input=train_file,
        epoch=25,  # Number of epochs
        lr=1.0,  # Learning rate
        wordNgrams=2,  # Use bigrams
        verbose=2,  # Verbose output
        loss='softmax'  # Loss function for classification
    )

    # Validate the model
    print("Evaluating model...")
    validation_result = model.test(val_file)
    print(f"Validation Results:\n"
          f" - Samples: {validation_result[0]}\n"
          f" - Precision: {validation_result[1]:.4f}\n"
          f" - Recall: {validation_result[2]:.4f}")

    # Save the model
    model.save_model(output_model)
    print(f"Model saved to {output_model}")
    return model


if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)

    # Split data for "birth year"
    split_data(birth_year_input_txt, birth_year_train_txt, birth_year_val_txt, birth_year_test_txt)

    # Split data for "political leaning"
    split_data(political_leaning_input_txt, political_leaning_train_txt, political_leaning_val_txt,
               political_leaning_test_txt)

    # Train FastText model for "birth year"
    birth_year_model = train_fasttext_model(
        birth_year_train_txt,
        birth_year_val_txt,
        r"C:\Users\Kaanncc\Desktop\birth_year_model.bin"
    )

    # Train FastText model for "political leaning"
    political_leaning_model = train_fasttext_model(
        political_leaning_train_txt,
        political_leaning_val_txt,
        r"C:\Users\Kaanncc\Desktop\political_leaning_model.bin"
    )
