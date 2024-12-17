import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.tag import pos_tag
from nltk.stem import WordNetLemmatizer
nltk.download(['punkt_tab', 'stopwords', 'averaged_perceptron_tagger_eng', 'wordnet'])

df_birth_year = pd.read_csv('datasets/birth_year.csv', sep=',')
df_political_leaning = pd.read_csv('datasets/political_leaning.csv', sep=',')

stop_words = set(stopwords.words("english")) - {'no', 'not', 'nor'}  # remove important negations from original list

lemmatizer = WordNetLemmatizer()

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

def tokenization(text: str) -> list[str]:
    """
    Tokenize the sentence.
    :param text: text string to tokenize.
    :return:
    """
    pos_text = pos_tag(word_tokenize(text))
    pos_tokens = [lemmatizer.lemmatize(word, pos_tagger(tag)).lower() for (word, tag) in pos_text]
    return pos_tokens

## example of output
df_by_h = df_birth_year.head().copy()
df_by_h['post'] = df_by_h['post'].apply(tokenization)
print(df_by_h)

## create new df and csv file of df_birth_year and df_political_leaning
# df_birth_year_tokenized = df_birth_year.copy()
# df_birth_year_tokenized['post'] = df_birth_year_tokenized['post'].apply(tokenization)
# df_birth_year_tokenized.to_csv('datasets/birth_year_tokenized.csv', index=False, header=True)
# print("CSV done")  # debug

# df_political_leaning_tokenized = df_political_leaning.copy()
# df_political_leaning_tokenized['post'] = df_political_leaning_tokenized['post'].apply(tokenization)
# df_political_leaning_tokenized.to_csv('datasets/political_leaning_tokenized.csv')
# print("CSV done")  # debug

