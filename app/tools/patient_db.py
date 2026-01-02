import json
from pathlib import Path
from functools import lru_cache

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "patients.json"

@lru_cache(maxsize=1)
def _load_patient_data():
    """Loads patient data from the JSON file and caches it."""
    with open(DATA_PATH) as f:
        return {p["patient_id"]: p for p in json.load(f)}

def get_patient_by_id(pid):
    """Retrieves a specific patient by ID."""
    data = _load_patient_data()
    if pid in data:
        return data[pid]
    raise ValueError("Patient not found")

def get_all_patient_ids():
    """Returns a list of all patient IDs in the database."""
    data = _load_patient_data()
    return list(data.keys())

def add_new_patients(new_patients: list):
    """Appends new patients to the JSON file if ID doesn't exist."""
    with open(DATA_PATH, 'r') as f:
        existing_data = json.load(f)
    
    existing_ids = {p["patient_id"] for p in existing_data}
    added_count = 0
    
    for p in new_patients:
        if "patient_id" in p and p["patient_id"] not in existing_ids:
            existing_data.append(p)
            added_count += 1
    
    if added_count > 0:
        with open(DATA_PATH, 'w') as f:
            json.dump(existing_data, f, indent=2)
        _load_patient_data.cache_clear()
    
    return added_count
