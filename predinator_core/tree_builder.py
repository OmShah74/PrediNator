import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
from .utils import MODEL_SAVE_PATH, METADATA_SAVE_PATH
from .data_manager import load_questions # For questions_map fallback
import time # For debug logging

class AkinatorTree:
    def __init__(self, ccp_alpha=0.0, max_depth=None, min_samples_leaf=1, min_samples_split=2): # MODIFIED PARAMS
        """
        Initialize the AkinatorTree.
        For initial testing with very small sample data (like from generate_sample_data.py):
        - ccp_alpha=0.0 (no pruning initially to ensure a tree can be built)
        - min_samples_leaf=1
        - min_samples_split=2
        Once you have a larger dataset (50-100+ characters), you can increase these and tune ccp_alpha.
        """
        print(f"[{time.ctime()}] TBUILDER: Initializing AkinatorTree with ccp_alpha={ccp_alpha}, "
              f"min_samples_leaf={min_samples_leaf}, min_samples_split={min_samples_split}")
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
        self.questions_map = {} # Stores Question objects, keyed by attribute_id

    def _prepare_data(self, df_celebs, questions_list):
        print(f"[{time.ctime()}] TBUILDER: _prepare_data called.")
        if df_celebs.empty or 'CelebrityName' not in df_celebs.columns:
            print(f"[{time.ctime()}] TBUILDER Error: Celebrity data is empty or 'CelebrityName' column is missing.")
            return None, None

        self.questions_map = {q.attribute_id: q for q in questions_list}
        self.feature_columns = [q.attribute_id for q in questions_list if q.attribute_id in df_celebs.columns]

        if not self.feature_columns:
            print(f"[{time.ctime()}] TBUILDER Error: No matching feature columns found between questions and celebrity data.")
            print(f"  Available celebrity columns: {df_celebs.columns.tolist()}")
            print(f"  Attribute IDs from questions: {[q.attribute_id for q in questions_list]}")
            return None, None
        print(f"[{time.ctime()}] TBUILDER: Using feature columns: {self.feature_columns}")

        X = df_celebs[self.feature_columns].copy()
        y_raw = df_celebs['CelebrityName']

        for col in X.columns:
            if X[col].dtype != float and X[col].dtype != np.float64:
                X[col] = pd.to_numeric(X[col], errors='coerce')

        if X.isnull().values.any():
            # This is expected if using DONT_KNOW_NUMERIC = np.nan
            nan_counts = X.isnull().sum()
            print(f"[{time.ctime()}] TBUILDER Note: NaN values present in features. Counts per column with NaNs:\n{nan_counts[nan_counts > 0]}")
        
        unique_labels = y_raw.unique()
        if len(unique_labels) < 2:
            print(f"[{time.ctime()}] TBUILDER Error: Need at least two unique celebrity names to train. Found {len(unique_labels)}: {unique_labels.tolist()}")
            return None, None
            
        y = self.label_encoder.fit_transform(y_raw)
        print(f"[{time.ctime()}] TBUILDER: Data prepared. X shape: {X.shape}, y shape: {y.shape}, Num unique labels: {len(np.unique(y))}")
        return X, y

    def train(self, df_celebs, questions_list):
        print(f"[{time.ctime()}] TBUILDER: train method called.")
        X, y = self._prepare_data(df_celebs, questions_list)

        if X is None or y is None or X.empty:
            print(f"[{time.ctime()}] TBUILDER: Training aborted due to data preparation issues.")
            self.model = None # Ensure model is None if training fails early
            return False
        
        if X.shape[0] == 0:
            print(f"[{time.ctime()}] TBUILDER Error: No samples available for training (X.shape[0] is 0).")
            self.model = None
            return False

        # min_samples_split check (already in DecisionTreeClassifier, but good to be aware)
        if X.shape[0] < self.model.min_samples_split:
             print(f"[{time.ctime()}] TBUILDER Warning: Number of samples ({X.shape[0]}) is less than min_samples_split ({self.model.min_samples_split}). "
                   "Training might still proceed but tree could be trivial.")
        
        # min_samples_leaf check (DecisionTreeClassifier will raise error if violated fundamentally)
        if X.shape[0] < self.model.min_samples_leaf * len(np.unique(y)) and len(np.unique(y)) > 1 : # Heuristic
            print(f"[{time.ctime()}] TBUILDER Warning: Number of samples ({X.shape[0]}) might be too low for min_samples_leaf ({self.model.min_samples_leaf}) "
                  f"across {len(np.unique(y))} classes. Tree might be trivial or fail.")


        print(f"[{time.ctime()}] TBUILDER: Attempting to fit model with {X.shape[0]} samples and {X.shape[1]} features.")
        try:
            self.model.fit(X, y) # This is where .tree_ gets created
            print(f"[{time.ctime()}] TBUILDER: Model training complete.")
            print(f"  Tree depth: {self.model.get_depth()}, Number of leaves: {self.model.get_n_leaves()}")
            if not hasattr(self.model, 'tree_') or self.model.tree_ is None:
                 print(f"[{time.ctime()}] TBUILDER CRITICAL ERROR: Model fitted but tree_ attribute is still missing or None!")
                 self.model = None # Mark as not properly trained
                 return False
            self.save_model_and_metadata()
            return True
        except ValueError as ve:
            print(f"[{time.ctime()}] TBUILDER ValueError during model.fit(): {ve}")
            print(f"  This often happens with insufficient samples for the given parameters "
                  f"(min_samples_split={self.model.min_samples_split}, min_samples_leaf={self.model.min_samples_leaf}).")
            print(f"  Consider reducing these parameters or adding more diverse data.")
            self.model = None # Ensure model is None on failure
            return False
        except Exception as e:
            print(f"[{time.ctime()}] TBUILDER Unexpected error during model.fit(): {e}")
            import traceback
            traceback.print_exc()
            self.model = None
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
            print(f"[{time.ctime()}] TBUILDER: Model saved to {MODEL_SAVE_PATH}")
            print(f"[{time.ctime()}] TBUILDER: Metadata saved to {METADATA_SAVE_PATH}")
        except Exception as e:
            print(f"[{time.ctime()}] TBUILDER Error saving model/metadata: {e}")


    def load_model_and_metadata(self):
        print(f"[{time.ctime()}] TBUILDER: Attempting to load model and metadata...")
        try:
            loaded_model_sklearn = joblib.load(MODEL_SAVE_PATH)
            # Crucial check: Ensure loaded object is a fitted tree
            if not hasattr(loaded_model_sklearn, 'tree_') or loaded_model_sklearn.tree_ is None:
                print(f"[{time.ctime()}] TBUILDER Error: Loaded model file exists but does not contain a fitted tree (no tree_ attribute). Treating as no model found.")
                self.model = None # Explicitly set model to None
                return False
            
            self.model = loaded_model_sklearn
            metadata = joblib.load(METADATA_SAVE_PATH)
            self.label_encoder = metadata['label_encoder']
            self.feature_columns = metadata['feature_columns']
            self.questions_map = metadata.get('questions_map', {})

            if not self.questions_map or len(self.questions_map) == 0: # Basic check for empty map
                print(f"[{time.ctime()}] TBUILDER Warning: questions_map from metadata is empty or missing. Reloading from questions.txt.")
                live_questions = load_questions()
                if live_questions:
                    self.questions_map = {q.attribute_id: q for q in live_questions}
                else:
                    print(f"[{time.ctime()}] TBUILDER Error: Failed to load questions from questions.txt for questions_map fallback.")
            
            print(f"[{time.ctime()}] TBUILDER: Model and metadata loaded successfully. Model is fitted: {hasattr(self.model, 'tree_') and self.model.tree_ is not None}")
            return True
        except FileNotFoundError:
            print(f"[{time.ctime()}] TBUILDER: No pre-trained model or metadata files found at {MODEL_SAVE_PATH} or {METADATA_SAVE_PATH}.")
            self.model = None
            return False
        except Exception as e:
            print(f"[{time.ctime()}] TBUILDER: Error loading model or metadata: {e}")
            import traceback
            traceback.print_exc()
            self.model = None
            return False

    def get_question_by_attribute_id(self, attr_id):
        return self.questions_map.get(attr_id)