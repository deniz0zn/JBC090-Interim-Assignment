import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
from nltk.tag import pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from tqdm import tqdm
nltk.download(['punkt_tab', 'stopwords', 'averaged_perceptron_tagger', 'wordnet', 'omw-1.4'])

df_birth_year = pd.read_csv('../datasets/birth_year.csv', sep=',')
df_political_leaning = pd.read_csv('../datasets/political_leaning.csv', sep=',')
lemmatizer = WordNetLemmatizer()
tqdm.pandas()  # add progress bar

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
        return wordnet.NOUN  # default POS is noun

stop_words = set(stopwords.words('english'))
def tokenization(text: str) -> list[str]:
    """
    Tokenize the sentence.
    :param text: text string to tokenize
    :return: list of tokens.
    """
    pos_text = pos_tag(word_tokenize(text))  # get POS for every word
    pos_tokens = [lemmatizer.lemmatize(word, pos_tagger(tag)).lower()
                  for (word, tag) in pos_text
                  if word.lower() not in stop_words]
    return pos_tokens


## create new df and save as parquet file
df_birth_year_tokenized = df_birth_year.copy()
df_birth_year_tokenized['post'] = df_birth_year_tokenized['post'].progress_apply(tokenization)
df_birth_year_tokenized.to_parquet('datasets/birth_year_tokenized.parquet')
print("'birth_year_tokenized.parquet' tokenized and saved.")

df_political_leaning_tokenized = df_political_leaning.copy()
df_political_leaning_tokenized['post'] = df_political_leaning_tokenized['post'].progress_apply(tokenization)
df_political_leaning_tokenized.to_parquet('datasets/political_leaning_tokenized.parquet')
print("'political_leaning_tokenized.parquet' tokenized and saved.")
