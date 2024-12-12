from dataloader import DataLoader
from ner_processor import NERProcessor


def main():
    data_loader = DataLoader()
    ner_processor = NERProcessor()

    # File paths
    input_path = r"C:\Users\Kaanncc\Desktop\birth_year.csv"  # Path to input CSV
    text_column = "post"
    output_path = r"C:\Users\Kaanncc\Desktop\birth_year_with_entities.csv"  # Path to save results

    print("Loading dataset...")
    df = data_loader.load_csv(input_path, required_columns=[text_column])

    print("Performing Named Entity Recognition...")
    processed_df = ner_processor.process_file(input_path, text_column, output_path)

    print("NER processing complete. Results saved.")
    print(processed_df.head())


if __name__ == "__main__":
    main()
