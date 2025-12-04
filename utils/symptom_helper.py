# This is a helper function that makes better use of the unique side effect codes provided by HODDI

import pandas as pd
import numpy as np
import os

# Define the location of the mapping CSV
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Update this path if your side effects CSV is not here
SIDE_EFFECTS_MAPPING_PATH = os.path.join(ROOT_DIR, "dictionary/Side_effects_unique.csv")

# Load the mapping table once
try:
    MAPPING_DF = pd.read_csv(SIDE_EFFECTS_MAPPING_PATH)
    # Filter the embedding columns for safety, they start at column index 2 (0, 1, 2, ...)
    EMBEDDING_DIMENSIONS = [col for col in MAPPING_DF.columns if col.isdigit()]
    EMBEDDING_DIM = len(EMBEDDING_DIMENSIONS)
except FileNotFoundError:
    print(f"Error: Mapping CSV not found at {SIDE_EFFECTS_MAPPING_PATH}")
    MAPPING_DF = None
    EMBEDDING_DIM = 128  # Defaulting to 128 which generates a 128-dimensional mean embedding vector from a list of user-provided symptoms


def generate_symptom_vector(user_symptoms_list: list, mapping_df: pd.DataFrame) -> np.ndarray:
    if mapping_df is None or not user_symptoms_list:
        return np.zeros(EMBEDDING_DIM, dtype=np.float32)

    # 1. Identify the embedding columns
    vector_cols = EMBEDDING_DIMENSIONS

    # 2. Find rows in the mapping DF that match the user's symptoms
    # Uses the column 'side_effect_name' in the csv for lookup
    symptom_vectors = mapping_df[mapping_df['side_effect_name'].isin(user_symptoms_list)]

    if symptom_vectors.empty:
        # If no symptom is found, return a vector of zeros (no symptom influence)
        return np.zeros(EMBEDDING_DIM, dtype=np.float32)

    # 3. Calculate the mean of the vectors
    mean_vector = symptom_vectors[vector_cols].mean(axis=0).to_numpy(dtype=np.float32)

    return mean_vector