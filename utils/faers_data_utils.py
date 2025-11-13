#Step 1: This step serves to download the data source to be inspected and to unzip it.

import requests
import os
import xml.etree.ElementTree as ET
import pandas as pd
import glob
import subprocess

BASE_URL = "https://fis.fda.gov/content/Exports/"

def fetch_and_extract_faers_xml(year: int, quarter: int, output_dir: str = "dataset"):
    if quarter not in (1, 2, 3, 4):
        raise ValueError("Quarter must be 1, 2, 3, or 4")
    if year < 2012:
        raise ValueError("Year must be >= 2012 for XML availability")

    # Ensure output directory exists, else list it is unavailable
    os.makedirs(output_dir, exist_ok=True)
    # This is inputting the name as the input year and quarter for the name of the file.
    filename = f"faers_xml_{year}q{quarter}.zip"
    #Showing where the zip is downloaded
    zip_path = os.path.join(output_dir, filename)
    # This is the formula FAERS website uses for their naming conventions
    url = BASE_URL + filename

    # Download the zip file
    response = requests.get(url, stream=True)
    if response.status_code != 200:
        raise FileNotFoundError(f"Could not retrieve file from: {url}")
    # Demonstrating to the user that it is hunting for a specific file
    print(f"Downloading {filename}...")
    with open(zip_path, 'wb') as f:
        for chunk in response.iter_content(1024):
            f.write(chunk)

    # Extract the ZIP from the downloaded site

    print("Extracting...")
    extract_dir = os.path.join(output_dir, f"faers_{year}q{quarter}")
    os.makedirs(extract_dir, exist_ok=True)

    # Use 7zip to extract Deflate64-compressed zips, this is essential because deflate64 wasn't possible for Mac
    subprocess.run(["7z", "x", zip_path, f"-o{extract_dir}"], check=True)

    # Parse each XML found, making a matrix for the list
    records = []
    # Going through each xml file in the directory where the data was extracted
    for xml_file in glob.glob(os.path.join(extract_dir, "*.xml")):
        # Showing the tree of the filesystem
        tree = ET.parse(xml_file)
        root = tree.getroot()
        xml_files = glob.glob(os.path.join(extract_dir, "**", "*.xml"), recursive=True)

        records = []
        for xml_file in xml_files:
            print(f"Processing {xml_file}...")
            tree = ET.parse(xml_file)
            root = tree.getroot()
            data = []

            for report in root.findall("safetyreport"):
                report_id = report.findtext("safetyreportid")
                patient = report.find("patient")
                if patient is None:
                    continue

                age = patient.findtext("patientonsetage")
                sex = patient.findtext("patientsex")

                reactions = patient.findall("reaction")
                for d in patient.findall("drug"):
                    drug = d.findtext("medicinalproduct")
                    substance = d.findtext("activesubstance/activesubstancename")
                    indication = d.findtext("drugindication")

                    for r in reactions:
                        reaction_term = r.findtext("reactionmeddrapt")
                        data.append({
                            "report_id": report_id,
                            "age": float(age) if age else None,
                            "sex": sex,
                            "drug": drug,
                            "substance": substance,
                            "indication": indication,
                            "reaction": reaction_term
                        })


year = int(input('Year of data to be used: '))
quarter = int(input("Quarter (1, 2, 3, or 4): "))
fetch_and_extract_faers_xml(year, quarter)

