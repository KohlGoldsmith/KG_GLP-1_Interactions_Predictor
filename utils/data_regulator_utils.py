# Step 2

# The python file takes an XML file and parses it by putting it into a csv and json file format
import xml.etree.ElementTree as ET
import pandas as pd

# Step 1: Parse XML
file_path = '/Users/kohlgoldsmith/PycharmProjects/Kohl_Goldsmith_GLP-1_Interactions_Predictor/dataset/faers_data/faers_xml_2025q3/XML/3_ADR25Q3.xml'
tree = ET.parse(file_path)
root = tree.getroot()

# Step 2: Extract data
data = []
for report in root.findall("safetyreport"):
    report_id = report.findtext("safetyreportid")
    patient = report.find("patient")
    if patient is None:
        continue

    age = patient.findtext("patientonsetage")
    sex = patient.findtext("patientsex")

    for d in patient.findall("drug"):
        drug = d.findtext("medicinalproduct")
        substance = d.findtext("activesubstance/activesubstancename") if d.find("activesubstance/activesubstancename") is not None else None
        indication = d.findtext("drugindication")
        reactions = patient.findall("reaction")
        for r in reactions:
            reaction_term = r.findtext("reactionmeddrapt")
            data.append({
                "report_id": report_id,
                "age": float(age) if age is not None else None,
                "sex": sex,
                "drug": drug,
                "substance": substance,
                "indication": indication,
                "reaction": reaction_term
            })

# Step 3: Create DataFrame
faers_df = pd.DataFrame(data)

# Step 4: Fill missing values
faers_df["drug"] = faers_df["drug"].fillna("Unknown")
faers_df["substance"] = faers_df["substance"].fillna("Unknown")
faers_df["indication"] = faers_df["indication"].fillna("Unknown")
faers_df["age"] = pd.to_numeric(faers_df["age"], errors="coerce").fillna(faers_df["age"].mean())
faers_df["sex"] = faers_df["sex"].fillna(faers_df["sex"].mode()[0])

# Step 5: Group by report_id
grouped_df = faers_df.groupby("report_id").agg({
    "age": "first",  # Keep patient age
    "sex": "first",  # Keep patient sex
    "drug": lambda x: list(x.dropna().unique()),
    "substance": lambda x: list(x.dropna().unique()),
    "indication": lambda x: list(x.dropna().unique()),
    "reaction": lambda x: list(x.dropna().unique())
}).reset_index()

# Step 6: Save grouped/cleaned data
output_csv = "/Users/kohlgoldsmith/PycharmProjects/Kohl_Goldsmith_GLP-1_Interactions_Predictor/dataset/faers_data/cleaned_data/grouped_cleaned_data.csv"
grouped_df.to_csv(output_csv, index=False)

output_json = "/Users/kohlgoldsmith/PycharmProjects/Kohl_Goldsmith_GLP-1_Interactions_Predictor/dataset/faers_data/cleaned_data/grouped_cleaned_data.json"
grouped_df.to_json(output_json, orient="records", lines=True)

# Step 7: Preview
print(grouped_df.head)
