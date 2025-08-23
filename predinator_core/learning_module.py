import pandas as pd
import numpy as np
from .data_manager import (load_celebrity_data, save_celebrity_data,
                           load_questions, save_questions, Question)
from .tree_builder import AkinatorTree
from .utils import answer_to_numeric, YES_NUMERIC, NO_NUMERIC, DONT_KNOW_NUMERIC

class LearningModule:
    def __init__(self, tree_handler: AkinatorTree):
        self.tree_handler = tree_handler
        self._refresh_all_questions_from_file() # Load all system-known questions

    def _refresh_all_questions_from_file(self):
        """Loads all questions from the questions.txt file."""
        self.all_questions_list = load_questions()
        # Optionally, update tree_handler's internal map if it's out of sync.
        # However, tree_handler.questions_map should reflect questions used for training.
        # For learning new celebs, we use all_questions_list as the master list.

    def learn_new_celebrity(self, actual_celebrity_name, game_path):
        print(f"\nLearning about new character: {actual_celebrity_name}")
        df_celebs = load_celebrity_data()
        self._refresh_all_questions_from_file() # Get latest questions

        if actual_celebrity_name in df_celebs['CelebrityName'].values:
            print(f"{actual_celebrity_name} already exists.")
            # Optional: "Would you like to update its attributes? (y/n)"
            # For now, we just return.
            return False

        new_celeb_attrs = {'CelebrityName': actual_celebrity_name}
        game_attributes_answered = {item['attribute_id']: item['answer'] for item in game_path}

        # Populate with answers from the game path
        for attr_id, answer in game_attributes_answered.items():
            new_celeb_attrs[attr_id] = answer

        # Ask for attributes not covered in game_path, using all_questions_list
        print(f"Please answer the remaining questions for {actual_celebrity_name} (yes/no/dont know or y/n/d/dk):")
        for q_obj in self.all_questions_list:
            attr_id = q_obj.attribute_id
            if attr_id not in game_attributes_answered: # If not answered during game
                while True:
                    # Use q_obj.possible_answers to show options, though we parse y/n/d
                    ans_str = input(f"Q: {q_obj.text} ({','.join(q_obj.possible_answers)}): ").strip().lower()
                    num_ans = answer_to_numeric(ans_str)
                    if num_ans is not None:
                        new_celeb_attrs[attr_id] = num_ans
                        break
                    else:
                        print("Invalid input. Please use 'yes', 'no', or 'dont know'.")
            # Ensure all attributes from all_questions_list are in new_celeb_attrs, defaulting to DONT_KNOW if missing
            if attr_id not in new_celeb_attrs:
                 new_celeb_attrs[attr_id] = DONT_KNOW_NUMERIC

        # Create a DataFrame for the new celebrity
        # Ensure it has all columns that df_celebs has, plus any new ones from all_questions_list
        all_known_attr_ids = {q.attribute_id for q in self.all_questions_list}
        
        # Prepare new row data, ensuring all attributes from all_questions_list are present
        final_new_celeb_row = {'CelebrityName': actual_celebrity_name}
        for attr_id_master in all_known_attr_ids:
            final_new_celeb_row[attr_id_master] = new_celeb_attrs.get(attr_id_master, DONT_KNOW_NUMERIC)
        
        new_row_df = pd.DataFrame([final_new_celeb_row])

        # Align columns with df_celebs and add new columns if any
        # Add missing columns to df_celebs (filled with DONT_KNOW_NUMERIC)
        for col in new_row_df.columns:
            if col != 'CelebrityName' and col not in df_celebs.columns:
                df_celebs[col] = DONT_KNOW_NUMERIC
        
        # Add missing columns to new_row_df (filled with DONT_KNOW_NUMERIC) - should be covered by above loop
        for col in df_celebs.columns:
            if col != 'CelebrityName' and col not in new_row_df.columns:
                new_row_df[col] = DONT_KNOW_NUMERIC
        
        # Concatenate, ensuring new_row_df has the same columns as df_celebs (or more if new questions added)
        df_celebs = pd.concat([df_celebs, new_row_df[df_celebs.columns.intersection(new_row_df.columns)]], ignore_index=True)
        
        # Ensure all attribute columns are float type
        for col in df_celebs.columns:
            if col != 'CelebrityName':
                df_celebs[col] = pd.to_numeric(df_celebs[col], errors='coerce')

        save_celebrity_data(df_celebs)
        print(f"'{actual_celebrity_name}' added to the dataset.")

        print("Retraining model with updated data (this may take a moment)...")
        # Pass the full, current list of questions for training consistency
        if self.tree_handler.train(df_celebs, self.all_questions_list):
            print("Model retrained successfully.")
            return True
        else:
            print("Failed to retrain model after adding new celebrity.")
            return False

    def add_new_question_and_learn(self, guessed_celebrity_name, actual_celebrity_name, game_path):
        print(f"\nMy guess ({guessed_celebrity_name or 'Unknown'}) was wrong. You were thinking of {actual_celebrity_name}.")
        print("Let's add a new distinguishing question.")
        self._refresh_all_questions_from_file() # Get current master list of questions

        while True:
            new_q_text = input("Enter the NEW question text (e.g., 'Is your character a Youtuber?'): ").strip()
            if new_q_text: break
            print("Question text cannot be empty.")

        existing_attr_ids = {q.attribute_id for q in self.all_questions_list}
        while True:
            new_q_attr_id = input("Enter a short, unique ID for this new question (e.g., 'is_youtuber'): ").strip().lower().replace(" ", "_")
            if new_q_attr_id and new_q_attr_id not in existing_attr_ids:
                break
            print("Attribute ID must be unique, not empty, and not already used.")

        def get_valid_answer_for_prompt(prompt_text):
            while True:
                ans_str = input(prompt_text).strip().lower()
                num_ans = answer_to_numeric(ans_str)
                if num_ans is not None: return num_ans
                print("Invalid input. Use 'yes', 'no', or 'dont know'.")

        ans_for_actual = get_valid_answer_for_prompt(f"Answer for '{actual_celebrity_name}' to '{new_q_text}' (y/n/dk): ")

        df_celebs = load_celebrity_data()
        ans_for_guessed = DONT_KNOW_NUMERIC # Default
        if guessed_celebrity_name and guessed_celebrity_name in df_celebs['CelebrityName'].values:
            ans_for_guessed = get_valid_answer_for_prompt(f"Answer for '{guessed_celebrity_name}' to '{new_q_text}' (y/n/dk): ")
        elif guessed_celebrity_name:
             print(f"Note: '{guessed_celebrity_name}' (my guess) was not found in the database. Will mark its answer as Don't Know for the new question.")


        # Add new question to the master list and save it
        new_question_obj = Question(new_q_attr_id, new_q_text, ['Yes', 'No', 'DontKnow'])
        self.all_questions_list.append(new_question_obj)
        save_questions(self.all_questions_list)
        print(f"New question '{new_q_text}' (ID: {new_q_attr_id}) added to questions.txt.")

        # Update the DataFrame: add new column, set values for guessed/actual, others DONT_KNOW
        df_celebs[new_q_attr_id] = DONT_KNOW_NUMERIC # Initialize for all
        df_celebs[new_q_attr_id] = df_celebs[new_q_attr_id].astype(float) # Ensure float type

        if guessed_celebrity_name and guessed_celebrity_name in df_celebs['CelebrityName'].values:
            df_celebs.loc[df_celebs['CelebrityName'] == guessed_celebrity_name, new_q_attr_id] = ans_for_guessed
        
        # The actual celebrity is not yet in df_celebs, so their answer for new_q_attr_id
        # will be handled by learn_new_celebrity.
        # We need to add this new attribute and its answer to the game_path for the actual celebrity.
        game_path_extended = game_path + [{'attribute_id': new_q_attr_id, 'answer': ans_for_actual}]

        save_celebrity_data(df_celebs) # Save changes to existing celebs for the new question
        print(f"Dataset updated with new attribute column '{new_q_attr_id}'.")

        # Now learn the actual celebrity, which will use the extended game_path
        # and will be prompted for any other questions from all_questions_list.
        return self.learn_new_celebrity(actual_celebrity_name, game_path_extended)