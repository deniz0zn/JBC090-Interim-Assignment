import pandas as pd

df_political_leaning_tokenized = pd.read_parquet('../datasets/political_leaning_tokenized.parquet')
df_political_leaning_tokenized_string = df_political_leaning_tokenized.copy()

df_political_leaning_tokenized_string['post'] = df_political_leaning_tokenized_string['post'].str.join(' ')

political_str_1 = r'\bcommunist\b|\bmarxist\b|\btrotskyist\b|\bstalinist\b|\bmaoist\b|\bleninist\b|\bneo-marxist\b'
political_str_2 = r'\bsocial democrat\b|\bdemocratic socialist\b|\beco-socialist\b'
political_str_3 = r'\bgreen\b|\bpopulist\b|\bnationalist\b|\bauthoritarian\b|\bfascist\b|\breactionary\b|\bradical\b'
political_str_4 = r'\bneoconservative\b|\bpaleoconservative\b|\banarchist\b|\banarcho-capitalist\b|\banarcho-communist\b'
political_str_5 = r'\balt-lite\b|\balt-left\b|\bpaleolibertarian\b|\bminarchist\b|\bclassical liberal\b'
political_str_6 = r'\bchristian democrat\b|\bchristian conservative\b|\bcenter-right\b|\bcenter-left\b'
political_str_7 = r'\bradical left\b|\bradical right\b|\bleft wing\b|\bleft-wing\b|\bright wing\b|\bright-wing\b'

# Combine all
political_regex = fr'{political_str_1}|{political_str_2}|{political_str_3}|{political_str_4}|{political_str_5}|{political_str_6}|{political_str_7}'

# Apply regex
df_political_leaning_cleaned = df_political_leaning_tokenized_string.copy()
df_political_leaning_cleaned['post'] = df_political_leaning_cleaned['post'].str.replace(political_regex, '<POLITICAL_TERM>', regex=True)

# Convertion to tokenized format
df_political_leaning_cleaned['post'] = df_political_leaning_cleaned['post'].str.split()


df_political_leaning_cleaned.to_parquet('datasets/political_leaning_tokenized_cleaned.parquet')

print("'political_leaning_tokenized_cleaned.parquet' saved successfully.")
