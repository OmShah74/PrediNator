# PREDINATOR/predinator_core/utils.py
import os
import numpy as np
import pandas as pd # Keep for numeric_to_answer or other pd uses
from django.conf import settings as django_settings

BASE_DIR_PROJECT_ROOT = str(django_settings.BASE_DIR)
DATA_DIR = os.path.join(BASE_DIR_PROJECT_ROOT, 'data')
MODEL_DIR = os.path.join(DATA_DIR, 'model')
QUESTIONS_FILE = os.path.join(DATA_DIR, 'questions.txt')
CELEBRITIES_FILE = os.path.join(DATA_DIR, 'celebrities.parquet')
MODEL_SAVE_PATH = os.path.join(MODEL_DIR, 'akinator_model.joblib')
METADATA_SAVE_PATH = os.path.join(MODEL_DIR, 'akinator_metadata.joblib')

try:
    os.makedirs(MODEL_DIR, exist_ok=True)
except Exception as e:
    print(f"Warning (predinator_core.utils): Could not create MODEL_DIR {MODEL_DIR} on import: {e}")

# These are standard Python floats and np.nan, which are handled correctly by the session logic now.
YES_NUMERIC = 1.0
NO_NUMERIC = 0.0
DONT_KNOW_NUMERIC = np.nan # JSON will store this as 'null'

def answer_to_numeric(answer_str):
    answer = str(answer_str).strip().lower()
    if answer in ['yes', 'y']:
        return YES_NUMERIC
    elif answer in ['no', 'n']:
        return NO_NUMERIC
    elif answer in ["don't know", "dont know", "d", "dk", "idk"]:
        return DONT_KNOW_NUMERIC
    return None # Unrecognized format

def numeric_to_answer(numeric_val):
    if pd.isna(numeric_val): # pd.isna handles None and np.nan
        return "Don't Know"
    if numeric_val == YES_NUMERIC:
        return 'Yes'
    if numeric_val == NO_NUMERIC:
        return 'No'
    return 'Unknown Value'