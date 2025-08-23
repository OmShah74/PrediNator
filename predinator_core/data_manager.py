import pandas as pd
import numpy as np
from .utils import (QUESTIONS_FILE, CELEBRITIES_FILE,
                    YES_NUMERIC, NO_NUMERIC, DONT_KNOW_NUMERIC)

class Question:
    def __init__(self, attribute_id, text, possible_answers):
        self.attribute_id = attribute_id
        self.text = text
        self.possible_answers = [pa.strip().lower() for pa in possible_answers]

    def __repr__(self):
        return f"Question(id='{self.attribute_id}', text='{self.text}')"

def load_questions():
    questions = []
    try:
        with open(QUESTIONS_FILE, 'r', encoding='utf-8') as f: # Added encoding
            header = next(f).strip()
            if header != "attribute_id::question_text::possible_answers":
                print("Warning: Questions file header mismatch or missing.")
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                try:
                    attr_id, text, answers_str = line.split('::')
                    possible_answers = [ans.strip() for ans in answers_str.split(',')]
                    questions.append(Question(attr_id.strip(), text.strip(), possible_answers))
                except ValueError:
                    print(f"Warning: Skipping malformed line in questions.txt: {line}")
    except FileNotFoundError:
        print(f"Error: {QUESTIONS_FILE} not found. Please run generate_sample_data.py")
        return []
    return questions

def load_celebrity_data():
    try:
        df = pd.read_parquet(CELEBRITIES_FILE, engine='pyarrow')
        if 'CelebrityName' not in df.columns:
            print("Error: 'CelebrityName' column missing in celebrities.parquet.")
            return pd.DataFrame()
        loaded_attribute_cols = [col for col in df.columns if col != 'CelebrityName']
        for col in loaded_attribute_cols:
            if df[col].dtype != float and df[col].dtype != np.float64:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        return df
    except FileNotFoundError:
        print(f"Error: {CELEBRITIES_FILE} not found. Please run generate_sample_data.py")
        return pd.DataFrame()
    except Exception as e:
        print(f"Error loading {CELEBRITIES_FILE}: {e}")
        return pd.DataFrame()

def save_celebrity_data(df):
    try:
        for col in df.columns:
            if col != 'CelebrityName':
                if df[col].dtype != float and df[col].dtype != np.float64:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
        df.to_parquet(CELEBRITIES_FILE, index=False, engine='pyarrow')
        print(f"Celebrity data saved to {CELEBRITIES_FILE}")
    except Exception as e:
        print(f"Error saving celebrity data to Parquet: {e}")

def save_questions(questions_list):
    try:
        with open(QUESTIONS_FILE, 'w', encoding='utf-8') as f: # Added encoding
            f.write("attribute_id::question_text::possible_answers\n")
            for q_obj in questions_list:
                # Ensure original case for 'DontKnow' if desired, or keep all lower
                answers_str = ",".join(pa.capitalize() if pa == "dontknow" else pa.capitalize() for pa in q_obj.possible_answers)
                f.write(f"{q_obj.attribute_id}::{q_obj.text}::{answers_str}\n")
        print(f"Questions saved to {QUESTIONS_FILE}")
    except Exception as e:
        print(f"Error saving questions: {e}")