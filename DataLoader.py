import pandas as pd


class DataLoader:
    def __init__(self):
        """
        Initialize the DataLoader for loading datasets.
        """
        pass

    @staticmethod
    def load_csv(file_path, required_columns=None):
        """
        Load a CSV file into a pandas DataFrame.

        Args:
            file_path (str): Path to the input CSV file.
            required_columns (list, optional): List of column names to validate in the file.

        Returns:
            pd.DataFrame: Loaded DataFrame.
        """
        # Load the CSV file
        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            raise ValueError(f"Error loading file at {file_path}: {e}")

        # Validate required columns
        if required_columns:
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns in file: {missing_columns}")

        print(f"Loaded dataset from {file_path}. Preview:")
        print(df.head())
        return df

    @staticmethod
    def save_csv(df, output_path):
        """
        Save a pandas DataFrame to a CSV file.

        Args:
            df (pd.DataFrame): DataFrame to save.
            output_path (str): Path to save the CSV file.
        """
        try:
            df.to_csv(output_path, index=False)
            print(f"File saved to: {output_path}")
        except Exception as e:
            raise ValueError(f"Error saving file to {output_path}: {e}")
