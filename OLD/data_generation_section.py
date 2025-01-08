import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt

class DataGenerationApplication:
    """
    Add column to dataframe that gives the generation of the birth year.
    :attributes: file_path, folder_path
    """
    def __init__(self, file_path: str, folder_path: str) -> None:
        """
        Initializes DataGenerationApplication class.
        :param file_path: path to birth_year.csv
        :param folder_path: folder in which cleaned files will be stored
        """
        self.df = pd.read_csv(f"{file_path}", sep=',')
        self.folder = folder_path

    def gen(self, year: int) -> str:
        """
        Determine generation of birth year.
        :param year: year to determine generation
        :return: generation.
        """
        if 1946 <= year <= 1964:
            return 'Baby boomers'
        elif 1965 <= year <= 1980:
            return 'Generation X'
        elif 1981 <= year <= 1996:
            return 'Millennials'
        else:  # 1997 <= year <= 2012
            return 'Generation Z'

    def apply_gen(self) -> pd.DataFrame:
        """
        Apply new column with generation of each birth year.
        :return: dataframe with new column.
        """
        self.df['generation'] = self.df['birth_year'].apply(self.gen)
        return self.df


class DataSection:
    """
    Data extraction used for writing the data section of the report.
    """
    def __init__(self, df: pd.DataFrame, target: str) -> None:
        self.df = df
        self.target = target

    def __str__(self):
        if self.target == 'birth_year':
            return (f"This dataset contains:\n{len(self.df)} rows\n"
                f"{len(self.df[self.target].unique())} unique values in '{self.target}' "
                f"ranging from {min(self.df[self.target].unique())} to {max(self.df[self.target].unique())}\n"
                f"(Most common value, rows): {Counter(self.df[self.target]).most_common()[0]} "
                f"and (least common value, rows): {Counter(self.df[self.target]).most_common()[-1]}")
        else:  # self.target == 'political leaning'
            return (f"This dataset contains:\n{len(self.df)} rows\n"
                    f"Value counts for each value: {Counter(self.df[self.target]).most_common()}\n")

    def visualize_birth_years(self) -> None:
        """
        Show that distribution of (un)grouped rows follow a similar trend.
        Therefore, no grouping is performed.
        """
        df_grouped = self.df.groupby(["auhtor_ID", self.target], sort=False, as_index=False).agg({"post": " ".join})
        normalized = self.df[self.target].value_counts(normalize=True).sort_index()
        normalized_grouped = df_grouped[self.target].value_counts(normalize=True).sort_index()
        x, y, y_g = normalized.index, normalized.values, normalized_grouped.values
        plt.figure(figsize=(10, 5))
        plt.plot(x, y, marker='o', color='b', linewidth=1.5)
        plt.plot(x, y_g, marker='o', color='r', linewidth=1.5)
        plt.title('Birth Year Distribution of Grouped and Ungrouped Rows')
        plt.legend(['Ungrouped', 'Grouped'])
        plt.xlabel('Birth Year')
        plt.ylabel('Number of Rows (%)')
        plt.ylim(-0.001, 0.061)
        plt.xlim(1947.5, 2010.5)
        plt.show()


df_age = pd.read_parquet('../datasets/birth_year_tokenized_cleaned.parquet')
age = DataSection(df_age, 'birth_year')
print(age)
# age.visualize_birth_years()

# df_gen = DataGenerationApplication('datasets/birth_year.csv', 'new_datasets').apply_gen()
# # df_gen.to_parquet('lai-data/birth_year_with_generation.parquet')