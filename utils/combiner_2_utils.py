# Load the already-encoded file from previous step
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder
import numpy as np

final_df = pd.read_csv('/Users/kohlgoldsmith/PycharmProjects/Kohl_Goldsmith_GLP-1_Interactions_Predictor/dataset/faers_data/cleaned_data/faers_with_embeddings_ready.csv')
# Identify key column groups
embedding_cols = [col for col in final_df.columns if col.isdigit()]  # 768-dim reaction embeddings
drug_cols = [col for col in final_df.columns if col.startswith("drug_")]
sex_cols = [col for col in final_df.columns if col.startswith("sex_")]
numeric_cols = ["age"]
print('Identified')
# === 1. Aggregate numerical + embeddings (mean)
agg_mean = final_df.groupby("report_id")[embedding_cols + numeric_cols].mean()
print('agg_mean: ', agg_mean)
# === 2. Aggregate drugs (any patient who took that drug → 1)
agg_drugs = final_df.groupby("report_id")[drug_cols].max()
print('agg_drugs: ', agg_drugs)

# === 3. Aggregate sex (mode: if multiple, take the most common one)
agg_sex = final_df.groupby("report_id")[sex_cols].mean().round()  # usually same for all rows per patient
print('agg_sex: ', agg_sex)

# === 4. Combine everything
aggregated_df = pd.concat([agg_mean, agg_drugs, agg_sex], axis=1).reset_index()
print('agg_df: ', aggregated_df)

print("Aggregated dataset shape:", aggregated_df.shape)
print("Example columns:", aggregated_df.columns[:15].tolist())
print(aggregated_df.head())

# === 5. Save final ML-ready dataset ===
aggregated_df.to_csv("/Users/kohlgoldsmith/PycharmProjects/Kohl_Goldsmith_GLP-1_Interactions_Predictor/dataset/faers_data/cleaned_data/faers_patient_aggregated.csv", index=False)
