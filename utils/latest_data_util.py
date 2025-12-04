# Helper function designed to fetch the latest data available from FAERS so that new
# Information is available to a user
import requests
from datetime import datetime

BASE_URL="https://fis.fda.gov/content/Exports/"

def get_latest_faers_release(max_year_back=15):
    now = datetime.now()
    current_year = now.year
    current_quarter = (now.month - 1) // 3 + 1

    for y in range(current_year, current_year - max_year_back, -1):
        for q in (4, 3, 2, 1):
            # Only check up to current quarter for the current year
            if y == current_year and q > current_quarter:
                continue

            filename = f"faers_xml_{y}q{q}.zip"
            url = BASE_URL + filename

            response = requests.head(url)
            if response.status_code == 200:
                return y, q

    return None, None
