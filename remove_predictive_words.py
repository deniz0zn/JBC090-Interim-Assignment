def get_top_predictive_words(model_path, top_n=20):
    """
    Get the top predictive words for each label.
    :param model_path: Path to the trained FastText model.
    :param top_n: Number of top predictive words to retrieve.
    """
    model = fasttext.load_model(model_path)
    words, _ = model.get_words(include_freq=True)

    label_words = {}
    for label in model.labels:
        word_scores = []
        for word in words:
            vector = model.get_input_vector(model.get_word_id(word))
            word_scores.append((word, vector.dot(model.get_output_matrix()[model.get_labels().index(label)])))

        # Sort by scores and take top N
        word_scores.sort(key=lambda x: x[1], reverse=True)
        label_words[label] = word_scores[:top_n]

    return label_words


# Analyze the birth year model
birth_year_top_words = get_top_predictive_words(r"C:\Users\Kaanncc\Desktop\birth_year_model.bin")
political_leaning_top_words = get_top_predictive_words(r"C:\Users\Kaanncc\Desktop\political_leaning_model.bin")


def remove_predictive_words(input_file, output_file, predictive_words):
    """
    Removes predictive words from the dataset.
    :param input_file: Path to the dataset.
    :param output_file: Path to save the cleaned dataset.
    :param predictive_words: List of words to remove.
    """
    with open(input_file, "r", encoding="utf-8") as infile, open(output_file, "w", encoding="utf-8") as outfile:
        for line in infile:
            label, text = line.split(" ", 1)
            for word in predictive_words:
                text = text.replace(word, "")
            text = ' '.join(text.split())  # Remove extra spaces
            outfile.write(f"{label} {text}\n")

# Define predictive words to remove
predictive_words_birth_year = ["label_1990", "label_2000"]  # Example
predictive_words_political_leaning = ["label_left", "label_right"]  # Example

# Clean datasets
cleaned_birth_year_file = r"C:\Users\Kaanncc\Desktop\cleaned_birth_year.txt"
cleaned_political_leaning_file = r"C:\Users\Kaanncc\Desktop\cleaned_political_leaning.txt"

remove_predictive_words(birth_year_cleaned_file, cleaned_birth_year_file, predictive_words_birth_year)
remove_predictive_words(political_leaning_cleaned_file, cleaned_political_leaning_file, predictive_words_political_leaning)

