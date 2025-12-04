# This works to combine the cleaned CSV files from the previous step into a One
# Hot Encoder friendly format and prepare the data for ML algorithms.
# This is the final step of the preprocessing stage

import pandas as pd
import numpy as np
import ast
import os
from collections import Counter
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder


def safe_list(x):
    # Safely converts a string representation of a list into a Python list.
    if isinstance(x, list):
        return x
    if isinstance(x, str):
        try:
            val = ast.literal_eval(x)
            # Ensure the result is iterable, or wrap in a list if necessary
            return val if isinstance(val, list) else [val]
        except:
            return []
    # New handling for NaN/None values from the merge/combine step
    if pd.isna(x):
        return []
    return []


def combine_with_embedding(cleaned_df_path, side_effect_csv, output_dir, top_n_drugs=100, reduced_emb_dim=128):
    # Combines FAERS cleaned CSV with side effect embeddings, performs one-hot encoding
    # for drugs in the tables and user sex, and aggregates features by report_id.

    # 1. Load DataFrames and Standardize Columns

    print(f"Loading FAERS data from: {cleaned_df_path}")

    #This is to stop a D-type warning that was benign
    try:
        faers_df = pd.read_csv(cleaned_df_path, error_bad_lines=False, warn_bad_lines=False, low_memory=False)
    except TypeError:
        faers_df = pd.read_csv(cleaned_df_path, low_memory=False)

    print(f"Loading side effect embeddings from: {side_effect_csv}")
    side_effects_df = pd.read_csv(side_effect_csv)

    # Normalize side effect column names for merging
    if "umls_cui_from_meddra" in side_effects_df.columns:
        side_effects_df.rename(columns={"umls_cui_from_meddra": "reaction_code"}, inplace=True)


    # 2. Unify Reaction Columns into a single 'reaction' list column
    if "reaction" not in faers_df.columns:
        # Find all reaction columns (reaction_1, reaction_2, ...)
        reaction_cols = [col for col in faers_df.columns if col.lower().startswith("reaction_")]

        if reaction_cols:
            print(f"Unifying {len(reaction_cols)} 'reaction_' columns into a single 'reaction' list column.")

            # Use .stack() to put all non-NaN values into a series, then .groupby() and .apply(list)
            faers_df["reaction"] = faers_df[reaction_cols].astype(str).replace(['nan', ''], np.nan).stack().groupby(
                level=0).apply(list).reindex(faers_df.index, fill_value=[])

            # Drop the individual columns to save memory, which is very stretched by the size of this data.
            faers_df.drop(columns=reaction_cols, inplace=True)
        else:
            available_cols = faers_df.columns.tolist()
            raise KeyError(
                "KeyError: 'reaction'. The input FAERS data is missing the required 'reaction' column. "
                "The utility function could not find or rename a suitable reaction column. "
                f"Available columns are: {available_cols}"
            )

    # 3: Unify Drug Columns into a single 'drug' list column
    if "drug" not in faers_df.columns:

        # Find all drug/substance columns (drug_1, drug_2, substance_1, substance_2)
        drug_substance_cols = [
            col for col in faers_df.columns
            if col.lower().startswith("drug_") or col.lower().startswith("substance_")
        ]

        if drug_substance_cols:
            print(
                f"Unifying {len(drug_substance_cols)} 'drug_' and 'substance_' columns into a single 'drug' list column.")

            # Similar to reaction: stack non-NaN values, group by report_id, and convert to list
            faers_df["drug"] = faers_df[drug_substance_cols].astype(str).replace(['nan', ''], np.nan).stack().groupby(
                level=0).apply(list).reindex(faers_df.index, fill_value=[])

            # Drop the individual columns to save memory
            faers_df.drop(columns=drug_substance_cols, inplace=True)
        else:
            available_cols = faers_df.columns.tolist()
            raise KeyError(
                "KeyError: 'drug'. The input FAERS data is missing the required 'drug' column. "
                "The utility function could not find or rename a suitable drug column. "
                f"Available columns are: {available_cols}"
            )

    # 4. Prepare Reactions for embedding

    # Parse reaction list strings to lists
    faers_df["reaction"] = faers_df["reaction"].apply(safe_list)

    # Explode rows: one row per report/reaction pair, this is used in older pandas versions
    faers_df_exploded = faers_df.explode("reaction")

    # Merge on the reaction NAME (e.g., 'Asthenia') to get embedding columns
    merged_df = faers_df_exploded.merge(
        side_effects_df,
        left_on="reaction",
        right_on="side_effect_name",
        how="left"
    )

    # 5. Reduce Embedding Dimensions
    reduced_cols = [str(i) for i in range(reduced_emb_dim)]
    # Extract embedding matrix, filling NaNs (for non-matched reactions) with 0.0 for OHE
    embedding_matrix = merged_df[reduced_cols].fillna(0).to_numpy(dtype=np.float32)
    print(f"Extracted {len(embedding_matrix)} rows for embedding aggregation.")

    # 6. Prepare drug features with One Hot Encoder
    # Parse drug list strings to actual lists (re-using safe_list)
    merged_df["drug"] = merged_df["drug"].apply(safe_list)

    # b. Filter to top-N drugs
    all_drugs = [d for sublist in merged_df["drug"] for d in sublist]
    top_drugs = [d for d, _ in Counter(all_drugs).most_common(top_n_drugs)]

    def filter_top(x):
        return [d for d in x if d in top_drugs]

    merged_df["drug"] = merged_df["drug"].apply(filter_top)

    # 7. Multi-Label Binarization (OHE)
    mlb = MultiLabelBinarizer(classes=top_drugs)
    drug_matrix = mlb.fit_transform(merged_df["drug"])
    drug_cols = [f"drug_{d}" for d in mlb.classes_]

    # 8. Prepare Sex Feature (OHE)

    if "sex" not in merged_df.columns:
        merged_df["sex"] = "Unknown"

    ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)
    sex_matrix = ohe.fit_transform(merged_df[["sex"]])
    try:
        sex_cols = ohe.get_feature_names_out(["sex"])
    except AttributeError:
        sex_cols = ohe.get_feature_names(["sex"])

    # 9. Aggregate Features by Report ID (from exploded rows back to one row)
    report_ids = merged_df["report_id"].to_numpy()
    uniq_reports = np.unique(report_ids)

    # Initialize arrays for aggregated data
    agg_emb = np.zeros((len(uniq_reports), reduced_emb_dim), dtype=np.float32)
    agg_age = np.zeros(len(uniq_reports), dtype=np.float32)
    agg_drugs = np.zeros((len(uniq_reports), len(drug_cols)), dtype=np.float32)
    agg_sex = np.zeros((len(uniq_reports), len(sex_cols)), dtype=np.float32)

    print(f"Aggregating {len(report_ids)} rows into {len(uniq_reports)} unique reports...")

    # 10. Loop through unique reports to aggregate (mean for embeddings, max for drugs/sex) This takes some time
    for i, rep in enumerate(uniq_reports):
        mask = report_ids == rep
        # Embeddings: Mean of all symptom embeddings for this report
        agg_emb[i] = embedding_matrix[mask].mean(axis=0)
        # Age: Mean (usually same value across rows)
        agg_age[i] = merged_df.loc[mask, "age"].mean() if "age" in merged_df.columns else 0
        # Drug/Sex OHE
        agg_drugs[i] = drug_matrix[mask].max(axis=0)
        agg_sex[i] = np.rint(sex_matrix[mask].mean(axis=0))  # Round in case of slight variance

    # 11. Build and Save Final DataFrame
    final_df = pd.DataFrame({
        "report_id": uniq_reports,
        "age": agg_age,
    })

    # 12. Concatenate all features
    final_df = pd.concat([
        final_df,
        pd.DataFrame(agg_sex, columns=sex_cols),
        pd.DataFrame(agg_drugs, columns=drug_cols),
        pd.DataFrame(agg_emb, columns=[f"emb_{i}" for i in range(reduced_emb_dim)])
    ], axis=1)

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "faers_with_embeddings_ready.csv")
    final_df.to_csv(out_path, index=False)


    print(f"Final dataset shape: {final_df.shape}")
    print(f"Saved prepared ML file to: {out_path}")

    return out_path
