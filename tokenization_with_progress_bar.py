import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.tag import pos_tag
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm

# Download NLTK resources
nltk.download(['punkt', 'stopwords', 'averaged_perceptron_tagger', 'wordnet', 'omw-1.4'])

# Load datasets
df_birth_year = pd.read_csv(r"C:\Users\Kaanncc\Desktop\birth_year.csv", sep=',')
df_political_leaning = pd.read_csv(r"C:\Users\Kaanncc\Desktop\political_leaning.csv", sep=',')

# Setup stopwords and lemmatizer
stop_words = set(stopwords.words("english")) - {'no', 'not', 'nor'}  # Remove negations from original stopwords list
lemmatizer = WordNetLemmatizer()

# POS Tagger
def pos_tagger(t: str):
    """
    Convert wordnet POS to lemmatization-POS.
    :param t: POS tag to convert
    :return: POS tag for lemmatization.
    """
    if t.startswith('J'):
        return wordnet.ADJ
    elif t.startswith('V'):
        return wordnet.VERB
    elif t.startswith('N'):
        return wordnet.NOUN
    elif t.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN  # Default POS is noun

# Tokenization Function
def tokenization(text: str) -> list[str]:
    """
    Tokenize and lemmatize the sentence.
    :param text: text string to tokenize and lemmatize.
    :return: List of tokens.
    """
    pos_text = pos_tag(word_tokenize(text))
    pos_tokens = [lemmatizer.lemmatize(word, pos_tagger(tag)).lower() for (word, tag) in pos_text]
    return pos_tokens

# Add Progress Bar
tqdm.pandas()

# Process `df_birth_year` with a progress bar
print("Processing birth_year.csv with tokenization...")
df_birth_year_tokenized = df_birth_year.copy()
df_birth_year_tokenized['post'] = df_birth_year_tokenized['post'].progress_apply(tokenization)
df_birth_year_tokenized.to_csv(r'C:\Users\Kaanncc\Desktop\birth_year_tokenized.csv', index=False, header=True)
print("birth_year.csv tokenized and saved.")

# Process `df_political_leaning` with a progress bar
print("Processing political_leaning.csv with tokenization...")
df_political_leaning_tokenized = df_political_leaning.copy()
df_political_leaning_tokenized['post'] = df_political_leaning_tokenized['post'].progress_apply(tokenization)
df_political_leaning_tokenized.to_csv(r'C:\Users\Kaanncc\Desktop\political_leaning_tokenized.csv', index=False, header=True)
print("political_leaning.csv tokenized and saved.")
