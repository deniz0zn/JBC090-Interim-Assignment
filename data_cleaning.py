import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
from nltk.tag import pos_tag
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm
nltk.download(['punkt', 'stopwords', 'averaged_perceptron_tagger', 'wordnet', 'omw-1.4'])

class DataCleaning:
    """
    Tokenize and clean the needed datasets.
    :attributes: path_to_birth_y, path_to_pol_lean, folder
    """
    def __init__(self, path_to_birth_y: str, path_to_pol_lean: str, folder: str) -> None:
        """
        Initializes DataCleaning class.
        :param path_to_birth_y: path to birth_year.csv
        :param path_to_pol_lean: path to political_leaning.csv
        :param folder: folder in which cleaned files will be stored
        """
        self.df_birth_y = pd.read_csv(f"{path_to_birth_y}", sep=',')
        self.df_pol_lean = pd.read_csv(f"{path_to_pol_lean}", sep=',')
        self.folder = folder
        self.lemmatizer = WordNetLemmatizer()

    @staticmethod
    def _pos_tagger(t: str) -> str:
        """
        Convert wordnet POS to lemmatization POS.
        :param t: POS tag to convert
        :return: POS tag for lemmatization.
        """
        return {'J': wordnet.ADJ, 'V': wordnet.VERB, 'R': wordnet.ADV}.get(t[0], wordnet.NOUN)

    def tokenize(self, text: str) -> list[str]:
        """
        Tokenize the sentence.
        :param text: text string to tokenize
        :return: list of tokens.
        """
        pos_text = pos_tag(word_tokenize(text))  # get POS for every word
        pos_tokens = [self.lemmatizer.lemmatize(word, self._pos_tagger(tag)).lower() for (word, tag) in pos_text]
        return pos_tokens

    def apply_tokenizer(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Apply tokenization to dataframe.
        :return: tokenized dataframes.
        """
        tqdm.pandas()  # show progress for each dataset
        self.df_birth_y['post'] = self.df_birth_y['post'].progress_apply(self.tokenize)
        self.df_pol_lean['post'] = self.df_pol_lean['post'].progress_apply(self.tokenize)
        return self.df_birth_y, self.df_pol_lean

    def clean(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Remove data pollution from dataframes e.g. ages and political terms.
        :return: cleaned dataframes.
        """
        self.df_birth_y['post'] = self.df_birth_y['post'].str.join(' ')
        self.df_pol_lean['post'] = self.df_pol_lean['post'].str.join(' ')

        age_str_1 = "\(?\s*\d+[MmFf]\s*\)?"  # both e.g. (31M) and 31M removed
        age_str_2 = "\(?\s*[MmFf]\d+\s*\)?"  # both e.g. (M31) and M31 removed
        age_str_4 = "\(\s*\d+(?:s| now)?\s*\)"  # all e.g. (31), (30s) and (31 now) removed
        age_str_3 = "[Ii]( \\'m| am)(?:\s+(?:almost|only|just|also|about to turn|now|a))?\s+\d+(?:\s+(?:year old|now))?"
        age_regex = fr'{age_str_1}|{age_str_2}|{age_str_3}|{age_str_4}'
        political_str_1 = r'\bcommunist\b|\bmarxist\b|\btrotskyist\b|\bstalinist\b|\bmaoist\b|\bleninist\b|\bneo-marxist\b'
        political_str_2 = r'\bsocial democrat\b|\bdemocratic socialist\b|\beco-socialist\b'
        political_str_3 = r'\bgreen\b|\bpopulist\b|\bnationalist\b|\bauthoritarian\b|\bfascist\b|\breactionary\b|\bradical\b'
        political_str_4 = r'\bneoconservative\b|\bpaleoconservative\b|\banarchist\b|\banarcho-capitalist\b|\banarcho-communist\b'
        political_str_5 = r'\balt-lite\b|\balt-left\b|\bpaleolibertarian\b|\bminarchist\b|\bclassical liberal\b'
        political_str_6 = r'\bchristian democrat\b|\bchristian conservative\b|\bcenter-right\b|\bcenter-left\b'
        political_str_7 = r'\bradical left\b|\bradical right\b|\bleft wing\b|\bleft-wing\b|\bright wing\b|\bright-wing\b'
        political_regex = fr'{political_str_1}|{political_str_2}|{political_str_3}|{political_str_4}|{political_str_5}|{political_str_6}|{political_str_7}'

        self.df_birth_y['post'] = self.df_birth_y['post'].str.replace(age_regex, '', regex=True)  # remove data pollution
        self.df_pol_lean['post'] = self.df_pol_lean['post'].str.replace(political_regex, '<POLITICAL_TERM>', regex=True)

        self.df_birth_y.to_parquet(f"{self.folder}/df_birth_y.parquet")  # save cleaned dfs
        self.df_pol_lean.to_parquet(f"{self.folder}/df_pol_lean.parquet")

        return self.df_birth_y, self.df_pol_lean


cleaner = DataCleaning('datasets/birth_year.csv', 'datasets/political_leaning.csv', 'new_datasets')
print("Tokenizing the dataframes:")
df_birth_year_tokenized, df_political_leaning_tokenized = cleaner.apply_tokenizer()
print("Dataframes tokenized. Cleaning the dataframes...")
df_birth_year, df_political_leaning = cleaner.clean()
print("Dataframes cleaned and saved.")
