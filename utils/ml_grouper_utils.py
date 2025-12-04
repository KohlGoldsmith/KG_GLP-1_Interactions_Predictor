#Step 5 Flattening the data to be usable

import pandas as pd
import ast
from pathlib import Path

def flatten_grouped_lists(grouped_csv_path: str, output_dir: str) -> str:
    print ("\nBeginning step 3: Flattening grouped lists")
    grouped_df = pd.read_csv(grouped_csv_path)

    # Convert list-like strings to readable lists
    list_columns = ["drug", "substance", "indication", "reaction"]
    for col in list_columns:
        grouped_df[col] = grouped_df[col].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) else []
        )

    # Compute maximum lengths
    max_len = {
        col: grouped_df[col].apply(len).max()
        for col in list_columns
    }

    flattened_df = grouped_df.copy()

    # Expand each list column into numbered columns
    for col in list_columns:
        for i in range(max_len[col]):
            flattened_df[f"{col}_{i+1}"] = flattened_df[col].apply(
                lambda lst, idx=i: lst[idx] if idx < len(lst) else None
            )

    # Drop original list columns
    flattened_df = flattened_df.drop(columns=list_columns)

    # Ensure output directory
    output_path_dir = Path(output_dir)
    output_path_dir.mkdir(parents=True, exist_ok=True)

    # Save files
    output_csv = str(output_path_dir / "flattened_cleaned_data.csv")
    output_json = str(output_path_dir / "flattened_cleaned_data.json")

    flattened_df.to_csv(output_csv, index=False)
    flattened_df.to_json(output_json, orient="records", lines=True)

    print("\nStep 3: List flattening completed.")
    print(flattened_df.head())

    return output_csv
