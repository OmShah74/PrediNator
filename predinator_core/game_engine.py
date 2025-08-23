# PREDINATOR/predinator_core/game_engine.py
import numpy as np
import pandas as pd
from .tree_builder import AkinatorTree
from .utils import answer_to_numeric
from .data_manager import load_questions, load_celebrity_data
import time

class GameEngine:
    def __init__(self):
        print(f"[{time.ctime()}] GAME_ENGINE: Initializing GameEngine instance (constructor)...")
        self.tree_handler = AkinatorTree() # Uses relaxed params if that's the default now
        self.game_active = False
        self.current_node_id = 0
        self.path_taken = [] # Path for the current game being processed by THIS engine instance
        # Initial model check. start_new_game will do a more thorough one.
        if not self._ensure_model_loaded_or_trained_internal():
            print(f"[{time.ctime()}] GAME_ENGINE Init: Initial model check failed. "
                  "Model will be loaded/trained on first game start demand.")

    def _ensure_model_loaded_or_trained_internal(self):
        """Internal helper to load or train model if not already valid. Returns True on success."""
        if self.tree_handler.model and hasattr(self.tree_handler.model, 'tree_') and self.tree_handler.model.tree_ is not None:
            # print(f"[{time.ctime()}] GAME_ENGINE _ensure_model: Model already loaded and seems fitted.")
            return True

        print(f"[{time.ctime()}] GAME_ENGINE _ensure_model: Model not present or not fitted in tree_handler. Attempting load/train...")
        if not self.tree_handler.load_model_and_metadata():
            print(f"[{time.ctime()}] GAME_ENGINE _ensure_model: Existing model load failed or model was not fitted. Attempting to train...")
            celebs_df = load_celebrity_data()
            questions_list = load_questions()
            if celebs_df.empty or not questions_list:
                print(f"[{time.ctime()}] GAME_ENGINE _ensure_model Error: Insufficient data for training (celebs or questions empty).")
                return False
            if not self.tree_handler.train(celebs_df, questions_list):
                print(f"[{time.ctime()}] GAME_ENGINE _ensure_model Error: Training new model FAILED.")
                return False
            print(f"[{time.ctime()}] GAME_ENGINE _ensure_model: New model trained successfully.")
        else:
            print(f"[{time.ctime()}] GAME_ENGINE _ensure_model: Model loaded successfully from file and appears fitted.")
        
        if not self.tree_handler.model or not hasattr(self.tree_handler.model, 'tree_') or self.tree_handler.model.tree_ is None:
             print(f"[{time.ctime()}] GAME_ENGINE _ensure_model CRITICAL: Model is STILL unavailable or not fitted after all attempts.")
             return False
        print(f"[{time.ctime()}] GAME_ENGINE _ensure_model: Model ensured successfully.")
        return True


    def start_new_game(self):
        print(f"[{time.ctime()}] GAME_ENGINE: start_new_game called for this engine instance.")
        
        if not self._ensure_model_loaded_or_trained_internal(): # Make sure model is ready
            print(f"[{time.ctime()}] GAME_ENGINE Error: Model could not be ensured for new game. Cannot start.")
            self.game_active = False # Set engine to inactive
            return False # Critical failure

        # Reset game state for this engine instance (session helpers will copy to/from session)
        self.current_node_id = 0
        self.path_taken = [] 
        self.game_active = True # CRITICAL: Game is now active FOR THIS ENGINE
        print(f"[{time.ctime()}] GAME_ENGINE: New game state initialized FOR ENGINE. Node: {self.current_node_id}, PathLen: {len(self.path_taken)}, Active: {self.game_active}")
        return True


    def get_next_question(self):
        # print(f"[{time.ctime()}] GAME_ENGINE: get_next_question. Engine Active: {self.game_active}, Engine Node: {self.current_node_id}")
        if not self.game_active: 
            # print(f"[{time.ctime()}] GAME_ENGINE: get_next_question - ENGINE game not active, returning leaf.")
            return None, True 
        
        # Model check has to be thorough
        if not self.tree_handler.model or \
           not hasattr(self.tree_handler.model, 'tree_') or \
           self.tree_handler.model.tree_ is None:
            print(f"[{time.ctime()}] GAME_ENGINE Error: get_next_question - Model is not trained/loaded (no tree_ attribute).")
            self.game_active = False
            return None, True

        tree_structure = self.tree_handler.model.tree_
        if not (0 <= self.current_node_id < len(tree_structure.feature)):
            print(f"[{time.ctime()}] GAME_ENGINE Error: current_node_id {self.current_node_id} out of bounds for tree.feature (len: {len(tree_structure.feature)}).")
            self.game_active = False; return None, True

        if tree_structure.feature[self.current_node_id] == -2: # Leaf node
            # print(f"[{time.ctime()}] GAME_ENGINE: At leaf node {self.current_node_id}.")
            return None, True # Indicates a leaf

        feature_idx = tree_structure.feature[self.current_node_id]
        try:
            if not (0 <= feature_idx < len(self.tree_handler.feature_columns)):
                print(f"[{time.ctime()}] GAME_ENGINE Error: feature_idx {feature_idx} out of bounds for feature_columns (len: {len(self.tree_handler.feature_columns)}).")
                self.game_active = False; return None, True
            attribute_id = self.tree_handler.feature_columns[feature_idx]
            question_obj = self.tree_handler.get_question_by_attribute_id(attribute_id)
            if question_obj:
                return question_obj, False # Question found, not a leaf
            else: # Should not happen if feature_columns and questions_map are synced
                print(f"[{time.ctime()}] GAME_ENGINE Error: No question object found for attr_id '{attribute_id}'.")
                self.game_active = False; return None, True
        except IndexError: 
            print(f"[{time.ctime()}] GAME_ENGINE IndexError in get_next_question (feature_idx: {feature_idx}).")
            self.game_active = False; return None, True
        except Exception as e:
            print(f"[{time.ctime()}] GAME_ENGINE Unexpected Error in get_next_question: {e}")
            self.game_active = False; return None, True


    def process_answer(self, answer_str):
        # print(f"[{time.ctime()}] GAME_ENGINE: process_answer with '{answer_str}'. Engine Active: {self.game_active}, Engine Node: {self.current_node_id}")
        if not self.game_active:
            print(f"[{time.ctime()}] GAME_ENGINE: process_answer - ENGINE game not active.")
            return False
        
        if not self.tree_handler.model or not hasattr(self.tree_handler.model, 'tree_') or self.tree_handler.model.tree_ is None:
             print(f"[{time.ctime()}] GAME_ENGINE Error: process_answer - Model not trained/loaded.")
             self.game_active = False; return False

        tree_structure = self.tree_handler.model.tree_
        if not (0 <= self.current_node_id < len(tree_structure.feature)) or \
           tree_structure.feature[self.current_node_id] == -2: # Already at a leaf
            print(f"[{time.ctime()}] GAME_ENGINE Warning: process_answer - at leaf or invalid node {self.current_node_id}. Cannot process further.")
            return False # Cannot process further if already at a conclusion point for questions

        feature_idx = tree_structure.feature[self.current_node_id]
        attribute_id = self.tree_handler.feature_columns[feature_idx]
        
        numeric_ans = answer_to_numeric(answer_str)
        if numeric_ans is None:
            print(f"[{time.ctime()}] GAME_ENGINE: Invalid answer string '{answer_str}' resulted in None numeric value.")
            return False # Indicate invalid answer format, let view handle message

        # Append to this engine instance's path_taken. Session helper will save it.
        self.path_taken.append({'attribute_id': attribute_id, 'answer': numeric_ans})
        
        prev_node = self.current_node_id
        threshold = tree_structure.threshold[self.current_node_id]
        left_child = tree_structure.children_left[self.current_node_id]
        right_child = tree_structure.children_right[self.current_node_id]
        # print(f"[{time.ctime()}] GAME_ENGINE: process_answer - Node {prev_node}, Thr {threshold:.2f}, LChild {left_child}, RChild {right_child}")


        if pd.isna(numeric_ans): # If answer is Don't Know (np.nan)
            self.current_node_id = right_child # Default scikit-learn behavior if not specially handled
            # print(f"[{time.ctime()}] GAME_ENGINE: Answer Don't Know (NaN), going RIGHT from {prev_node} to {self.current_node_id}.")
        elif numeric_ans <= threshold:
            self.current_node_id = left_child
            # print(f"[{time.ctime()}] GAME_ENGINE: Answer '{numeric_ans}', going LEFT from {prev_node} to {self.current_node_id}.")
        else: 
            self.current_node_id = right_child
            # print(f"[{time.ctime()}] GAME_ENGINE: Answer '{numeric_ans}', going RIGHT from {prev_node} to {self.current_node_id}.")
        
        # print(f"[{time.ctime()}] GAME_ENGINE: Path taken updated by engine: {self.path_taken[-1] if self.path_taken else 'Path empty'}")
        return True

    def make_guess(self):
        # print(f"[{time.ctime()}] GAME_ENGINE: make_guess called. Engine Active: {self.game_active}, Engine Node: {self.current_node_id}")
        if not self.game_active: # Game might have become inactive due to an error before reaching guess
            print(f"[{time.ctime()}] GAME_ENGINE: make_guess - ENGINE game already inactive.")
            return None
        
        if not self.tree_handler.model or not hasattr(self.tree_handler.model, 'tree_') or self.tree_handler.model.tree_ is None \
           or not self.tree_handler.label_encoder.classes_.size: # Check label_encoder too
             print(f"[{time.ctime()}] GAME_ENGINE Error: make_guess - Model not trained/loaded or label_encoder empty.")
             self.game_active = False; return None

        tree_structure = self.tree_handler.model.tree_
        if not (0 <= self.current_node_id < len(tree_structure.feature)) or \
           tree_structure.feature[self.current_node_id] != -2: # Must be a leaf
            print(f"[{time.ctime()}] GAME_ENGINE Error: Not at a valid leaf node ({self.current_node_id}) to make a guess.")
            self.game_active = False; return None

        class_counts = tree_structure.value[self.current_node_id][0]
        predicted_class_index = np.argmax(class_counts)
        total_samples_at_leaf = np.sum(class_counts)
        confidence = (np.max(class_counts) / total_samples_at_leaf) if total_samples_at_leaf > 0 else 0.0
        
        try:
            guessed_celebrity_name = self.tree_handler.label_encoder.inverse_transform([predicted_class_index])[0]
            print(f"[{time.ctime()}] GAME_ENGINE: Guessing '{guessed_celebrity_name}' (Conf: {confidence:.2f}) from node {self.current_node_id}")
            self.game_active = False # Guess made (or attempted), this game round in engine ends
            return guessed_celebrity_name
        except Exception as e:
            print(f"[{time.ctime()}] GAME_ENGINE Error during guess decoding: {e}")
            self.game_active = False; return None