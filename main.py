# Main file that runs all  utils to fetch the latest data from FAERS and regulate it for the model

from utils.latest_data_util import get_latest_faers_release
from utils.faers_data_utils import fetch_and_extract_faers_xml
from utils.data_regulator_utils import regulate_data
from utils.combiner_utils import combine_with_embedding
from utils.ml_grouper_utils import flatten_grouped_lists
import pandas as pd

#Showing Pandas version for issues troubleshooting with implementations.
print("Pandas Version: ", pd.__version__)

def main():
    print ("Checking latest FAERS data release...")
    latest_year, latest_quarter = get_latest_faers_release()
    if latest_year:
        print(f"Latest FAERS XML available: {latest_year} Q{latest_quarter}")
    else:
        print("\nCould not detect latest FAERS release.")

# Users choose which dataset to pull from the website
    print("\nPlease enter which dataset you want to use.")
    year = int(input("Enter FAERS Year: "))
    quarter = int(input("Enter Quarter (1–4): "))

    # STEP 1
    extracted_dir, extracted_data = fetch_and_extract_faers_xml(year, quarter)

    # STEP 2
    cleaned_csv = regulate_data(extract_dir=extracted_dir, output_dir="dataset/faers_data/cleaned_data")
    
    # STEP 3
    ready_for_combine_csv = flatten_grouped_lists(
        grouped_csv_path=cleaned_csv,
        output_dir="dataset/faers_data/cleaned_data"
    )
    # STEP 4
    combine_with_embedding(
        ready_for_combine_csv,  # Pass the CSV that has the parsed 'drug' and 'reaction' lists
        "dictionary/Side_effects_unique.csv",
        "dataset/processed"
    )

if __name__ == "__main__":
    main()
