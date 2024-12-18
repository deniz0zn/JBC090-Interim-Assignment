import pandas as pd
from tqdm import tqdm

# File paths for tokenized CSVs and output FastText files
birth_year_csv = r"C:\Users\Kaanncc\Desktop\birth_year_tokenized.csv"
political_leaning_csv = r"C:\Users\Kaanncc\Desktop\political_leaning_tokenized.csv"

birth_year_output_txt = r"C:\Users\Kaanncc\Desktop\birth_year_fasttext.txt"
political_leaning_output_txt = r"C:\Users\Kaanncc\Desktop\political_leaning_fasttext.txt"

# Load data
df_birth_year = pd.read_csv(birth_year_csv)
df_political_leaning = pd.read_csv(political_leaning_csv)

# FastText conversion function
def convert_to_fasttext(df, label_col, text_col, output_file):
    """Convert a DataFrame to FastText format with a progress bar."""
    with open(output_file, 'w', encoding='utf-8') as f_out:
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {output_file}"):
            label = f"__label__{row[label_col]}"
            text = " ".join(eval(row[text_col]))  # Assuming 'post' is a tokenized list stored as a string
            f_out.write(f"{label} {text}\n")

# Process birth_year dataset
print("Processing Birth Year Dataset...")
#convert_to_fasttext(
    #df=df_birth_year,
    #label_col='birth_year',
    #text_col='post',
    #output_file=birth_year_output_txt
#)
print(f"Birth Year Dataset converted and saved to {birth_year_output_txt}")

# Process political_leaning dataset
print("Processing Political Leaning Dataset...")
#convert_to_fasttext(
    #df=df_political_leaning,
    #label_col='political_leaning',
    #text_col='post',
    #output_file=political_leaning_output_txt
#)
print(f"Political Leaning Dataset converted and saved to {political_leaning_output_txt}")


# Define paths for previewing the generated FastText files
birth_year_output_txt = r"C:\Users\Kaanncc\Desktop\birth_year_fasttext.txt"
political_leaning_output_txt = r"C:\Users\Kaanncc\Desktop\political_leaning_fasttext.txt"

# Function to preview top N lines from a file
def preview_file(file_path, n=10):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = [file.readline().strip() for _ in range(n)]
    return lines

# Preview top 10 entries for both files
birth_year_preview = preview_file(birth_year_output_txt, 10)
political_leaning_preview = preview_file(political_leaning_output_txt, 10)

print(birth_year_preview, political_leaning_preview)
