import os
import urllib.request
import pandas as pd

DOWNLOAD_ROOT = "https://github.com/ageron/data/raw/main/"
LIFESAT_PATH = os.path.join("datasets")
LIFESAT_URL = DOWNLOAD_ROOT + "lifesat/lifesat.csv"


def fetch_lifesat_data(LIFESAT_url=LIFESAT_URL, LIFESAT_path=LIFESAT_PATH):
    if not os.path.isdir(LIFESAT_path):
        os.makedirs(LIFESAT_path)
    csv_path = os.path.join(LIFESAT_path, "LIFESAT.csv")
    urllib.request.urlretrieve(LIFESAT_url, csv_path)


def load_lifesat_data(LIFESAT_path=LIFESAT_PATH):
    csv_path = os.path.join(LIFESAT_path, "LIFESAT.csv")
    return pd.read_csv(csv_path)