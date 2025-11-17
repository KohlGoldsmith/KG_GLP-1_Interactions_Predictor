# Step 3: Flatten the data in preparation for ML usage

import pandas as pd
import ast
# Step 1: Load CSV
def csv_reader(cleaned_csv_path: str):
    print("Beginning step 3: Flattening CSV...")
    faers_df = pd.read_csv(cleaned_csv_path, on_bad_lines = 'skip')

# Step 2: Parse the 'safetyreport' column into Python objects
    faers_df['safetyreport_parsed'] = faers_df['ichicsr.safetyreport'].apply(lambda x: ast.literal_eval(x))

# Step 3: Explode the list of safety reports into separate rows
    df_exploded = faers_df.explode('safetyreport_parsed').reset_index(drop=True)

# Step 4: Flatten nested JSON into columns
    df_normalized = pd.json_normalize(df_exploded['safetyreport_parsed'])

# Step 5: Optionally keep some top-level CSV info too
    top_columns = ['ichicsr.ichicsrmessageheader.messagetype',
               'ichicsr.ichicsrmessageheader.messagenumb',
               'ichicsr.ichicsrmessageheader.messagesenderidentifier']
    df_final = pd.concat([df_exploded[top_columns], df_normalized], axis=1)

# Step 6: Print the resulting table nicely
    pd.set_option('display.max_columns', None)  # show all columns
    pd.set_option('display.max_rows', 20)       # show first 20 rows only
    pd.set_option('display.width', 200)         # adjust width for readability
    print(df_final.head())

# Step 7: Group reactions and indications back per safetyreport
    grouped = df_final.groupby("safetyreportid").agg({
        "medicinalproduct": lambda x: list(x.dropna().unique()),
        "indication": lambda x: list(x.dropna().unique()),
        "reaction": lambda x: list(x.dropna().unique())
    }).reset_index()
    print(df_final.columns.tolist())
    print(grouped.head())

    print("Step 3 complete: Flattened data above.")
    return df_final
