#Main file that runs all  utils to fetch the latest data from FAERS and regulate it for the model

from utils.latest_data_util import get_latest_faers_release   # if you put it in utils/latest_checker.py
from utils.faers_data_utils import fetch_and_extract_faers_xml
from utils.data_regulator_utils import regulate_data
from utils.csv_reader_utils import csv_reader
from utils.combiner_utils import combine_with_embedding
from utils.ml_grouper_utils import flatten_grouped_lists


def main():
    print ("Checking latest FAERS data release...")
    latest_year, latest_quarter = get_latest_faers_release()
    if latest_year:
        print(f"Latest FAERS XML available: {latest_year} Q{latest_quarter}")
    else:
        print("\nCould not detect latest FAERS release.")

    print("\nPlease enter which dataset you want to use.")
    year = int(input("Enter FAERS Year: "))
    quarter = int(input("Enter Quarter (1–4): "))

    # STEP 1
    extracted_dir = fetch_and_extract_faers_xml(year, quarter)

    # STEP 2
    cleaned_csv = regulate_data(extract_dir=extracted_dir, output_dir="dataset/faers_data/cleaned_data")
    # STEP 3
    #flattened_df, flattened_csv_path = csv_reader(cleaned_csv)

    # STEP 4
    final_csv = combine_with_embedding(
        cleaned_csv,
        "dataset/dictionary/Side_effects_unique.csv",
        "dataset/processed"
    )
    # STEP 5
    final_flattened_csv = flatten_grouped_lists(
        grouped_csv_path=final_csv,
        output_dir="dataset/faers_data/cleaned_data"
    )
    print("\nPipeline complete and data is prepared.")
    print("Final ML file:", final_flattened_csv)

if __name__ == "__main__":
    main()
