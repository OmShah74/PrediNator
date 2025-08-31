# PREDINATOR/predinator_core/tree_builder.py
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
from .utils import MODEL_SAVE_PATH, METADATA_SAVE_PATH
from .data_manager import load_questions
import time

class AkinatorTree:
    def __init__(self, ccp_alpha=0.0, max_depth=None, min_samples_leaf=1, min_samples_split=2):
        """
        Initializes the AkinatorTree handler.
        Note: The DecisionTreeClassifier is instantiated here, but it will be refitted during training.
        """
        print(f"[{time.ctime()}] TBUILDER: Initializing AkinatorTree instance.")
        self.model = DecisionTreeClassifier(
            criterion='gini',
            random_state=42,
            ccp_alpha=ccp_alpha,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            min_samples_split=min_samples_split
        )
        self.label_encoder = LabelEncoder()
        self.feature_columns = []
        self.questions_map = {}

    def _prepare_data(self, df_celebs, questions_list):
        print(f"[{time.ctime()}] TBUILDER: _prepare_data called.")
        if df_celebs.empty or 'CelebrityName' not in df_celebs.columns:
            print(f"[{time.ctime()}] TBUILDER Error: Celebrity data is empty or missing 'CelebrityName'.")
            return None, None

        self.questions_map = {q.attribute_id: q for q in questions_list}
        self.feature_columns = [q.attribute_id for q in questions_list if q.attribute_id in df_celebs.columns]

        if not self.feature_columns:
            print(f"[{time.ctime()}] TBUILDER Error: No matching feature columns found.")
            return None, None

        X = df_celebs[self.feature_columns].copy()
        y_raw = df_celebs['CelebrityName']

        # Ensure all feature columns are numeric, coercing errors
        for col in X.columns:
            X[col] = pd.to_numeric(X[col], errors='coerce')

        # Handle potential NaN values that might result from coercion or new questions
        if X.isnull().values.any():
            nan_counts = X.isnull().sum()
            print(f"[{time.ctime()}] TBUILDER Note: NaN values found. Filling with 0.0 (No). Columns:\n{nan_counts[nan_counts > 0]}")
            X = X.fillna(0.0) # Fill any remaining NaNs with 0.0 for robustness

        if len(y_raw.unique()) < 2:
            print(f"[{time.ctime()}] TBUILDER Error: Need at least two unique celebrities to train. Found {len(y_raw.unique())}.")
            return None, None
            
        y = self.label_encoder.fit_transform(y_raw)
        print(f"[{time.ctime()}] TBUILDER: Data prepared. X shape: {X.shape}, y shape: {y.shape}")
        return X, y

    def train(self, df_celebs, questions_list):
        print(f"[{time.ctime()}] TBUILDER: train method called.")
        X, y = self._prepare_data(df_celebs, questions_list)

        if X is None or y is None or X.empty:
            print(f"[{time.ctime()}] TBUILDER: Training aborted due to data preparation issues.")
            return False
        
        # Create a new, clean model instance for this training session.
        # This prevents any old state or invalid parameters from causing issues.
        new_model = DecisionTreeClassifier(
            criterion='gini',
            random_state=42,
            ccp_alpha=self.model.ccp_alpha,
            max_depth=self.model.max_depth,
            min_samples_leaf=self.model.min_samples_leaf,
            min_samples_split=self.model.min_samples_split
        )

        print(f"[{time.ctime()}] TBUILDER: Attempting to fit new model with {X.shape[0]} samples.")
        try:
            new_model.fit(X, y)
            print(f"[{time.ctime()}] TBUILDER: Model training complete.")
            
            # --- CRITICAL CHANGE ---
            # Only replace the live model if training was successful.
            self.model = new_model
            
            self.save_model_and_metadata()
            return True
        except Exception as e:
            print(f"[{time.ctime()}] TBUILDER CRITICAL ERROR during model.fit(): {e}")
            import traceback
            traceback.print_exc()
            # --- CRITICAL CHANGE ---
            # DO NOT set self.model to None. Leave the old, working model intact.
            print(f"[{time.ctime()}] TBUILDER: Training failed. The existing model will be kept active.")
            return False

    def save_model_and_metadata(self):
        print(f"[{time.ctime()}] TBUILDER: Saving model and metadata...")
        try:
            joblib.dump(self.model, MODEL_SAVE_PATH)
            metadata = {
                'label_encoder': self.label_encoder,
                'feature_columns': self.feature_columns,
                'questions_map': self.questions_map
            }
            joblib.dump(metadata, METADATA_SAVE_PATH)
            print(f"[{time.ctime()}] TBUILDER: Model and metadata saved successfully.")
        except Exception as e:
            print(f"[{time.ctime()}] TBUILDER Error saving model/metadata: {e}")

    def load_model_and_metadata(self):
        print(f"[{time.ctime()}] TBUILDER: Attempting to load model and metadata...")
        try:
            loaded_model = joblib.load(MODEL_SAVE_PATH)
            if not hasattr(loaded_model, 'tree_') or loaded_model.tree_ is None:
                print(f"[{time.ctime()}] TBUILDER Error: Loaded model file is not a fitted tree.")
                return False
            
            self.model = loaded_model
            metadata = joblib.load(METADATA_SAVE_PATH)
            self.label_encoder = metadata['label_encoder']
            self.feature_columns = metadata['feature_columns']
            self.questions_map = metadata.get('questions_map', {})
            
            print(f"[{time.ctime()}] TBUILDER: Model and metadata loaded successfully.")
            return True
        except FileNotFoundError:
            print(f"[{time.ctime()}] TBUILDER: No pre-trained model found. Please run train_model.py.")
            return False
        except Exception as e:
            print(f"[{time.ctime()}] TBUILDER: Error loading model or metadata: {e}")
            return False

    def get_question_by_attribute_id(self, attr_id):
        return self.questions_map.get(attr_id)