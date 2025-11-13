# Step 4 Combining the side effects with the cleaned data

import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder
import numpy as np

#1. Load data sources of the groueed data and the side effects list
faers_df = pd.read_csv("/Users/kohlgoldsmith/PycharmProjects/Kohl_Goldsmith_GLP-1_Interactions_Predictor/dataset/faers_data/cleaned_data/grouped_cleaned_data.csv")  # already flattened

side_effects_df = pd.read_csv("/Users/kohlgoldsmith/PycharmProjects/Kohl_Goldsmith_GLP-1_Interactions_Predictor/dataset/dictionary/Side_effects_unique.csv")

# 2. Rename the previously clunky umls to the reaction's code
side_effects_df.rename(columns={"umls_cui_from_meddra": "reaction_code"}, inplace=True)

# 3. Merge embeddings so that the reaction and reaction code are moved together
merged_df = faers_df.merge(
    side_effects_df, left_on="reaction", right_on="reaction_code", how="left"
)

# 4. Extract the embedding columns
embedding_cols = [str(i) for i in range(768)]  # Number of entries in the dictionary
embedding_matrix = merged_df[embedding_cols].fillna(0)

# 5. Clean the drug column since it's a list of multiple drugs
import ast
merged_df["drug"] = merged_df["drug"].apply(
    lambda x: ast.literal_eval(x) if isinstance(x, str) else x
)

# 6. One-hot encode the list of drugs so that 1 is present and 0 is absent
mlb = MultiLabelBinarizer()
drug_encoded = mlb.fit_transform(merged_df["drug"])
drug_encoded_df = pd.DataFrame(drug_encoded, columns=[f"drug_{d}" for d in mlb.classes_])

# 7. One-hot encode sex where sex_1 is Male and sex_2 is Female
ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
sex_encoded = ohe.fit_transform(merged_df[["sex"]])
sex_encoded_df = pd.DataFrame(sex_encoded, columns=ohe.get_feature_names_out(["sex"]))

# 8. Combine all data into a new dataframe
final_df = pd.concat(
    [
        merged_df[["report_id", "age"]],
        sex_encoded_df,
        drug_encoded_df,
        embedding_matrix,
    ],
    axis=1,
)

print("Final dataset head :")
print(final_df.head())

# 9. Save for modeling
final_df.to_csv("/Users/kohlgoldsmith/PycharmProjects/Kohl_Goldsmith_GLP-1_Interactions_Predictor/dataset/faers_data/cleaned_data/faers_with_embeddings_ready.csv", index=False)
print("Done saving")

# Load the already-encoded file from previous step

# Identify key column groups
embedding_cols = [col for col in final_df.columns if col.isdigit()]  # 768-dim reaction embeddings
drug_cols = [col for col in final_df.columns if col.startswith("drug_")]
sex_cols = [col for col in final_df.columns if col.startswith("sex_")]
numeric_cols = ["age"]  # Add other numeric features as needed

# === 1. Aggregate numerical + embeddings (mean)
agg_mean = final_df.groupby("report_id")[embedding_cols + numeric_cols].mean()

# === 2. Aggregate drugs (any patient who took that drug → 1)
agg_drugs = final_df.groupby("report_id")[drug_cols].max()

# === 3. Aggregate sex (mode: if multiple, take the most common one)
agg_sex = final_df.groupby("report_id")[sex_cols].mean().round()  # usually same for all rows per patient

# === 4. Combine everything
aggregated_df = pd.concat([agg_mean, agg_drugs, agg_sex], axis=1).reset_index()

print("Aggregated dataset shape:", aggregated_df.shape)
print("Example columns:", aggregated_df.columns[:15].tolist())
print(aggregated_df.head())

# === 5. Save final ML-ready dataset ===
aggregated_df.to_csv("/Users/kohlgoldsmith/PycharmProjects/Kohl_Goldsmith_GLP-1_Interactions_Predictor/dataset/faers_data/cleaned_data/faers_patient_aggregated.csv", index=False)
