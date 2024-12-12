import pandas as pd
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
from tqdm import tqdm
import torch


class NERProcessor:
    def __init__(self, model_name="dslim/bert-base-NER"):
        """
        Initialize the NERProcessor with a specified model.
        """
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(self.model_name).to(self.device)
        self.ner_pipeline = pipeline("ner", model=self.model, tokenizer=self.tokenizer,
                                     grouped_entities=True, device=0 if torch.cuda.is_available() else -1)

    def process_file(self, file_path, text_column, output_path):
        """
        Perform NER on a specified CSV file and save the results.

        Args:
            file_path (str): Path to the input CSV file.
            text_column (str): Column containing text data for NER.
            output_path (str): Path to save the processed CSV file.
        """
        # Load the dataset
        df = pd.read_csv(file_path)
        print(f"Loaded dataset from {file_path}. Preview:")
        print(df.head())

        # Ensure the text column exists
        if text_column not in df.columns:
            raise ValueError(f"Column '{text_column}' not found in the dataset.")

        # Extract text data
        text_data = df[text_column].dropna().tolist()

        # Perform NER with a progress bar
        ner_results = []
        for text in tqdm(text_data, desc=f"Processing NER for {file_path}"):
            ner_results.append(self.ner_pipeline(text))

        # Add the results to the DataFrame
        df['extracted_entities'] = ner_results

        # Save the updated DataFrame
        df.to_csv(output_path, index=False)
        print(f"NER results saved to: {output_path}")

        return df
