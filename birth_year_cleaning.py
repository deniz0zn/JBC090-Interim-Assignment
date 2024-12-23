import pandas as pd

df_birth_year_tokenized = pd.read_parquet('birth_year_tokenized.parquet')
df_birth_year_tokenized_string = df_birth_year_tokenized.copy()
df_birth_year_tokenized_string['post'] = df_birth_year_tokenized_string['post'].str.join(' ')

age_str_1 = "\(?\s*\d+[MmFf]\s*\)?"  # both e.g. (31M) and 31M removed
age_str_2 = "\(?\s*[MmFf]\d+\s*\)?"  # both e.g. (M31) and M31 removed
age_str_4 = "\(\s*\d+(?:s| now)?\s*\)"  # all e.g. (31), (30s) and (31 now) removed
age_str_3 = "[Ii]( \\'m| am)(?:\s+(?:almost|only|just|also|about to turn|now|a))?\s+\d+(?:\s+(?:year old|now))?"

df_birth_year_tokenized_cleaned = df_birth_year_tokenized_string.copy()
df_birth_year_tokenized_cleaned['post'] = df_birth_year_tokenized_cleaned['post'].str.replace(fr'{age_str_1}|{age_str_2}|{age_str_3}|{age_str_4}', '', regex=True)

df_birth_year_tokenized_cleaned.to_parquet('datasets/birth_year_tokenized_cleaned.parquet')
