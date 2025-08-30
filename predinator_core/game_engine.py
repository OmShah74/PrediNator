# PREDINATOR/predinator_core/game_engine.py
import numpy as np
import pandas as pd
from .tree_builder import AkinatorTree
from .utils import answer_to_numeric
import time

class GameEngine:
    def __init__(self):
        print(f"[{time.ctime()}] GAME_ENGINE: Initializing GameEngine instance...")
        self.tree_handler = AkinatorTree()
        self.game_active = False
        self.current_node_id = 0
        self.path_taken = []
        
        # On initialization, the engine MUST load a pre-trained model.
        if not self.tree_handler.load_model_and_metadata():
            # This is a critical failure. The application cannot run without a model.
            print(f"[{time.ctime()}] GAME_ENGINE CRITICAL ERROR: No pre-trained model found or failed to load.")
            print(f"[{time.ctime()}] Please run 'python train_model.py' to create the model files before starting the server.")
            # We don't raise an exception here to allow Django to start,
            # but the views will check and render an error page.
            self.tree_handler.model = None # Ensure model is None
        else:
            print(f"[{time.ctime()}] GAME_ENGINE: Pre-trained model loaded successfully.")

    def start_new_game(self):
        print(f"[{time.ctime()}] GAME_ENGINE: start_new_game called.")
        
        # Check if the model is valid before starting.
        if not self.tree_handler.model:
            print(f"[{time.ctime()}] GAME_ENGINE Error: Cannot start new game, model is not loaded.")
            self.game_active = False
            return False

        self.current_node_id = 0
        self.path_taken = [] 
        self.game_active = True
        print(f"[{time.ctime()}] GAME_ENGINE: New game state initialized. Active: {self.game_active}")
        return True

    def get_next_question(self):
        if not self.game_active or not self.tree_handler.model:
            return None, True 

        tree = self.tree_handler.model.tree_
        node_id = self.current_node_id

        if tree.feature[node_id] == -2: # Leaf node
            return None, True

        feature_idx = tree.feature[node_id]
        attribute_id = self.tree_handler.feature_columns[feature_idx]
        question_obj = self.tree_handler.get_question_by_attribute_id(attribute_id)

        if question_obj:
            return question_obj, False
        else:
            print(f"[{time.ctime()}] GAME_ENGINE Error: No question object found for attr_id '{attribute_id}'.")
            self.game_active = False
            return None, True

    def process_answer(self, answer_str):
        if not self.game_active or not self.tree_handler.model:
            return False
        
        tree = self.tree_handler.model.tree_
        node_id = self.current_node_id

        if tree.feature[node_id] == -2: # Already at a leaf
            return False

        numeric_ans = answer_to_numeric(answer_str)
        if numeric_ans is None:
            return False

        feature_idx = tree.feature[node_id]
        attribute_id = self.tree_handler.feature_columns[feature_idx]
        self.path_taken.append({'attribute_id': attribute_id, 'answer': numeric_ans})
        
        threshold = tree.threshold[node_id]

        if pd.isna(numeric_ans):
            # scikit-learn decision trees do not natively handle NaNs during prediction.
            # The convention is that they might go left or right. A common practice is to send them to the larger node.
            # However, for simplicity and consistency with how they might be handled in training, we'll send them right.
            self.current_node_id = tree.children_right[node_id]
        elif numeric_ans <= threshold:
            self.current_node_id = tree.children_left[node_id]
        else: 
            self.current_node_id = tree.children_right[node_id]
        
        return True

    def make_guess(self):
        if not self.game_active or not self.tree_handler.model:
            return None
        
        tree = self.tree_handler.model.tree_
        node_id = self.current_node_id

        if tree.feature[node_id] != -2: # Must be a leaf
            print(f"[{time.ctime()}] GAME_ENGINE Error: Not at a leaf node ({node_id}) to make a guess.")
            self.game_active = False
            return None

        class_counts = tree.value[node_id][0]
        predicted_class_index = np.argmax(class_counts)
        
        try:
            guessed_celebrity = self.tree_handler.label_encoder.inverse_transform([predicted_class_index])[0]
            self.game_active = False
            return guessed_celebrity
        except Exception as e:
            print(f"[{time.ctime()}] GAME_ENGINE Error during guess decoding: {e}")
            self.game_active = False
            return None