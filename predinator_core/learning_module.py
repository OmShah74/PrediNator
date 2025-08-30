# predinator/predinator_core/learning_module.py
import pandas as pd
import numpy as np
import json # For web context
import time # For logging
from .data_manager import (load_celebrity_data, save_celebrity_data,
                           load_questions, save_questions, Question)
from .tree_builder import AkinatorTree
from .utils import answer_to_numeric, DONT_KNOW_NUMERIC

class LearningModule:
    def __init__(self, tree_handler: AkinatorTree):
        self.tree_handler = tree_handler
        self._refresh_all_questions_from_file() # Load all system-known questions

    def _refresh_all_questions_from_file(self):
        """Loads all questions from the questions.txt file."""
        print(f"[{time.ctime()}] LEARNER: Refreshing all questions from file.")
        self.all_questions_list = load_questions()

    def learn_new_celebrity_fully_web(self, actual_celebrity_name, game_path_answers, all_submitted_attributes):
        """
        Web-specific function to learn a new celebrity from form data and retrain the model.
        - actual_celebrity_name: The name of the new character.
        - game_path_answers: Dict of {attr_id: numeric_answer} from the game path.
        - all_submitted_attributes: Dict of {attr_id: 'yes'/'no'/'dontknow'} from the full learn form.
        """
        print(f"[{time.ctime()}] LEARNER: learn_new_celebrity_fully_web called for '{actual_celebrity_name}'.")
        df_celebs = load_celebrity_data()
        self._refresh_all_questions_from_file()

        if actual_celebrity_name in df_celebs['CelebrityName'].values:
            print(f"[{time.ctime()}] LEARNER Warning: '{actual_celebrity_name}' already exists. Aborting learn process.")
            return False # Or handle as an update later

        new_celeb_attrs = {'CelebrityName': actual_celebrity_name}

        # Use the fully submitted attributes as the source of truth, converting them to numeric
        for attr_id, ans_str in all_submitted_attributes.items():
            new_celeb_attrs[attr_id] = answer_to_numeric(ans_str)

        # Ensure all known attributes are present in the new celebrity's data, defaulting to DONT_KNOW
        all_known_attr_ids = {q.attribute_id for q in self.all_questions_list}
        for attr_id_master in all_known_attr_ids:
            if attr_id_master not in new_celeb_attrs:
                new_celeb_attrs[attr_id_master] = DONT_KNOW_NUMERIC

        new_row_df = pd.DataFrame([new_celeb_attrs])

        # Align columns with the main dataframe before concatenating
        for col in df_celebs.columns:
            if col != 'CelebrityName' and col not in new_row_df.columns:
                new_row_df[col] = DONT_KNOW_NUMERIC
        
        df_celebs = pd.concat([df_celebs, new_row_df[df_celebs.columns]], ignore_index=True)

        for col in df_celebs.columns:
            if col != 'CelebrityName':
                df_celebs[col] = pd.to_numeric(df_celebs[col], errors='coerce')
        
        save_celebrity_data(df_celebs)
        print(f"[{time.ctime()}] LEARNER: '{actual_celebrity_name}' saved to dataset. Retraining model...")

        if self.tree_handler.train(df_celebs, self.all_questions_list):
            print(f"[{time.ctime()}] LEARNER: Model retrained successfully.")
            return True
        else:
            print(f"[{time.ctime()}] LEARNER Error: Failed to retrain model after adding new celebrity.")
            return False

    def web_add_question_and_learn_redirect(self, guessed_celebrity_name, actual_celebrity_name, game_path,
                                           new_q_text, new_q_attr_id, ans_for_actual_new_q_str, ans_for_guessed_new_q_str):
        """
        Web-specific function to add a new question and PREPARE for learning the new celebrity.
        This function does not do the final learning but prepares the context for the attribute collection view.
        """
        print(f"[{time.ctime()}] LEARNER: web_add_question_and_learn_redirect called for '{actual_celebrity_name}' with new question '{new_q_attr_id}'.")
        self._refresh_all_questions_from_file()
        existing_attr_ids = {q.attribute_id for q in self.all_questions_list}
        if new_q_attr_id in existing_attr_ids:
            print(f"[{time.ctime()}] LEARNER Error: Attribute ID '{new_q_attr_id}' already exists.")
            return None

        # Add new question to master list and save it
        new_question_obj = Question(new_q_attr_id, new_q_text, ['Yes', 'No', 'DontKnow'])
        self.all_questions_list.append(new_question_obj)
        save_questions(self.all_questions_list)
        print(f"[{time.ctime()}] LEARNER: New question saved to questions.txt.")

        # Update the DataFrame with the new question column
        df_celebs = load_celebrity_data()
        df_celebs[new_q_attr_id] = DONT_KNOW_NUMERIC
        df_celebs[new_q_attr_id] = df_celebs[new_q_attr_id].astype(float)

        # Update the answer for the guessed celebrity if they exist
        if guessed_celebrity_name and guessed_celebrity_name in df_celebs['CelebrityName'].values:
            ans_for_guessed_num = answer_to_numeric(ans_for_guessed_new_q_str)
            df_celebs.loc[df_celebs['CelebrityName'] == guessed_celebrity_name, new_q_attr_id] = ans_for_guessed_num
        
        save_celebrity_data(df_celebs)
        print(f"[{time.ctime()}] LEARNER: Celebrities dataset updated with new attribute column '{new_q_attr_id}'.")

        # Now, prepare the context for the learn_new_celebrity_attributes.html template
        game_path_answers = {item['attribute_id']: item['answer'] for item in game_path}
        
        # Add the answer for the new question for the NEW celebrity to this dictionary
        game_path_answers[new_q_attr_id] = answer_to_numeric(ans_for_actual_new_q_str)

        questions_to_ask_on_form = []
        for q_obj in self.all_questions_list: # Use the newly updated list
            current_answer = game_path_answers.get(q_obj.attribute_id)
            questions_to_ask_on_form.append({
                'attribute_id': q_obj.attribute_id,
                'text': q_obj.text,
                'current_answer': current_answer,
                'current_answer_is_nan': pd.isna(current_answer)
            })

        context_for_next_step = {
            'celebrity_name': actual_celebrity_name,
            'questions_to_ask': questions_to_ask_on_form,
            'game_path_json': json.dumps(game_path_answers), # This now includes the new question's answer
            'form_action': 'submit_new_celebrity_attributes',
            'new_question_info_json': json.dumps({ # Pass new question info along if needed
                'id': new_q_attr_id,
                'text': new_q_text
            })
        }
        return context_for_next_step