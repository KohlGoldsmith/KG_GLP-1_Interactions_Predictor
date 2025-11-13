#Step 5

import pandas as pd

# Load grouped data
grouped_df = pd.read_csv("/Users/kohlgoldsmith/PycharmProjects/Kohl_Goldsmith_GLP-1_Interactions_Predictor/dataset/faers_data/cleaned_data/grouped_cleaned_data.csv")
import pandas as pd

# Assume grouped_df is your grouped DataFrame from previous step

# Step 0: Replace any non-list entries with empty lists for safety
for col in ['drug','substance','indication','reaction']:
    grouped_df[col] = grouped_df[col].apply(lambda x: x if isinstance(x, list) else [])

# Step 1: Determine max lengths of lists in each column
max_drugs = grouped_df['drug'].apply(len).max()
max_substances = grouped_df['substance'].apply(len).max()
max_indications = grouped_df['indication'].apply(len).max()
max_reactions = grouped_df['reaction'].apply(len).max()

# Step 2: Flatten lists into separate columns safely
flattened_df = grouped_df.copy()

# Drugs
for i in range(max_drugs):
    flattened_df[f'drug_{i+1}'] = flattened_df['drug'].apply(lambda x: x[i] if i < len(x) else None)

# Substances
for i in range(max_substances):
    flattened_df[f'substance_{i+1}'] = flattened_df['substance'].apply(lambda x: x[i] if i < len(x) else None)

# Indications
for i in range(max_indications):
    flattened_df[f'indication_{i+1}'] = flattened_df['indication'].apply(lambda x: x[i] if i < len(x) else None)

# Reactions
for i in range(max_reactions):
    flattened_df[f'reaction_{i+1}'] = flattened_df['reaction'].apply(lambda x: x[i] if i < len(x) else None)

# Step 3: Drop original list columns
flattened_df = flattened_df.drop(columns=['drug','substance','indication','reaction'])

# Step 4: Save flattened data
output_csv = "/Users/kohlgoldsmith/PycharmProjects/Kohl_Goldsmith_GLP-1_Interactions_Predictor/dataset/faers_data/cleaned_data/flattened_cleaned_data.csv"
flattened_df.to_csv(output_csv, index=False)

output_json = "/Users/kohlgoldsmith/PycharmProjects/Kohl_Goldsmith_GLP-1_Interactions_Predictor/dataset/faers_data/cleaned_data/flattened_cleaned_data.json"
flattened_df.to_json(output_json, orient="records", lines=True)

# Step 5: Preview
print(flattened_df.head())
