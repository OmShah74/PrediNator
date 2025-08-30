import numpy as np
import pandas as pd
import os

DATA_DIR = 'data'
QUESTIONS_FILE_PATH = os.path.join(DATA_DIR, 'questions.txt')
CELEBRITIES_FILE_PATH = os.path.join(DATA_DIR, 'celebrities.parquet')

def generate_questions_file():
    """
    Generates the questions.txt file which serves as the master list of all
    possible questions the game can ask.
    """
    # The 'DontKnow' option is kept in the text file for the user interface,
    # but our dataset will not contain 'Don't Know' values for training.
    questions_content = """attribute_id::question_text::possible_answers
is_male::Is your character male?::Yes,No,DontKnow
is_female::Is your character female?::Yes,No,DontKnow
is_real_person::Is your character a real person (not fictional)?::Yes,No,DontKnow
is_fictional::Is your character fictional?::Yes,No,DontKnow
is_american::Is your character American?::Yes,No,DontKnow
is_european::Is your character European?::Yes,No,DontKnow
is_asian::Is your character Asian?::Yes,No,DontKnow
is_actor::Is your character primarily known as an actor/actress?::Yes,No,DontKnow
is_singer::Is your character primarily known as a singer?::Yes,No,DontKnow
is_musician::Is your character primarily known as a musician (plays an instrument)?::Yes,No,DontKnow
is_politician::Is your character a politician?::Yes,No,DontKnow
is_athlete::Is your character an athlete?::Yes,No,DontKnow
is_scientist::Is your character a scientist?::Yes,No,DontKnow
is_writer::Is your character a writer/author?::Yes,No,DontKnow
is_alive::Is your character currently alive?::Yes,No,DontKnow
died_before_2000::Did your character die before the year 2000?::Yes,No,DontKnow
born_after_1980::Was your character born after 1980?::Yes,No,DontKnow
has_dark_hair::Does your character usually have dark hair (black/brown)?::Yes,No,DontKnow
has_blonde_hair::Does your character usually have blonde hair?::Yes,No,DontKnow
has_red_hair::Does your character usually have red hair?::Yes,No,DontKnow
wears_glasses::Does your character often wear glasses?::Yes,No,DontKnow
has_facial_hair::Does your character often have facial hair (beard/mustache)?::Yes,No,DontKnow
won_oscar::Has your character won an Academy Award (Oscar)?::Yes,No,DontKnow
won_grammy::Has your character won a Grammy Award?::Yes,No,DontKnow
starred_in_marvel_movie::Has your character starred in a Marvel movie?::Yes,No,DontKnow
starred_in_dc_movie::Has your character starred in a DC movie?::Yes,No,DontKnow
known_for_comedy::Is your character known for comedy?::Yes,No,DontKnow
from_usa_movie::Is your character from a movie primarily made in the USA?::Yes,No,DontKnow
from_tv_show::Is your character primarily known from a TV show?::Yes,No,DontKnow
has_superpowers::Does your character have superpowers (if fictional)?::Yes,No,DontKnow
is_humanoid::Is your character humanoid (if fictional)?::Yes,No,DontKnow
is_animal::Is your character an animal (or animal-like)?::Yes,No,DontKnow
"""
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(QUESTIONS_FILE_PATH, 'w') as f:
        f.write(questions_content.strip())
    print(f"Generated {QUESTIONS_FILE_PATH}")

def generate_celebrities_parquet():
    """
    Generates the celebrities.parquet file.

    This function now ensures every character has a definitive 1.0 (Yes) or 0.0 (No)
    for every single question, creating a complete and consistent dataset
    perfect for training a high-quality decision tree.
    """
    # This dictionary is the source of truth. 1.0 means "Yes".
    # Any question not listed for a character will automatically be "No" (0.0).
    characters = {
        # Actors & Actresses
        'Tom Hanks': {'is_male': 1.0, 'is_real_person': 1.0, 'is_american': 1.0, 'is_actor': 1.0, 'is_alive': 1.0, 'won_oscar': 1.0, 'has_dark_hair': 1.0, 'known_for_comedy': 1.0, 'from_usa_movie': 1.0},
        'Scarlett Johansson': {'is_female': 1.0, 'is_real_person': 1.0, 'is_american': 1.0, 'is_actor': 1.0, 'is_alive': 1.0, 'starred_in_marvel_movie': 1.0, 'born_after_1980': 1.0, 'has_blonde_hair': 1.0, 'from_usa_movie': 1.0},
        'Leonardo DiCaprio': {'is_male': 1.0, 'is_real_person': 1.0, 'is_american': 1.0, 'is_actor': 1.0, 'is_alive': 1.0, 'won_oscar': 1.0, 'has_blonde_hair': 1.0, 'from_usa_movie': 1.0},
        'Meryl Streep': {'is_female': 1.0, 'is_real_person': 1.0, 'is_american': 1.0, 'is_actor': 1.0, 'is_alive': 1.0, 'won_oscar': 1.0, 'has_blonde_hair': 1.0, 'from_usa_movie': 1.0},
        'Will Smith': {'is_male': 1.0, 'is_real_person': 1.0, 'is_american': 1.0, 'is_actor': 1.0, 'is_alive': 1.0, 'known_for_comedy': 1.0, 'from_usa_movie': 1.0, 'is_singer': 1.0},
        'Robert Downey Jr.': {'is_male': 1.0, 'is_real_person': 1.0, 'is_american': 1.0, 'is_actor': 1.0, 'is_alive': 1.0, 'starred_in_marvel_movie': 1.0, 'has_dark_hair': 1.0, 'from_usa_movie': 1.0},
        'Keanu Reeves': {'is_male': 1.0, 'is_real_person': 1.0, 'is_actor': 1.0, 'is_alive': 1.0, 'has_dark_hair': 1.0, 'starred_in_dc_movie': 1.0, 'from_usa_movie': 1.0},
        'Dwayne Johnson': {'is_male': 1.0, 'is_real_person': 1.0, 'is_american': 1.0, 'is_actor': 1.0, 'is_alive': 1.0, 'is_athlete': 1.0, 'from_usa_movie': 1.0},

        # Musicians
        'Taylor Swift': {'is_female': 1.0, 'is_real_person': 1.0, 'is_american': 1.0, 'is_singer': 1.0, 'is_alive': 1.0, 'won_grammy': 1.0, 'born_after_1980': 1.0, 'has_blonde_hair': 1.0},
        'Beyoncé': {'is_female': 1.0, 'is_real_person': 1.0, 'is_american': 1.0, 'is_singer': 1.0, 'is_alive': 1.0, 'won_grammy': 1.0, 'has_dark_hair': 1.0, 'born_after_1980': 1.0},
        'Ed Sheeran': {'is_male': 1.0, 'is_real_person': 1.0, 'is_european': 1.0, 'is_singer': 1.0, 'is_alive': 1.0, 'won_grammy': 1.0, 'has_red_hair': 1.0, 'wears_glasses': 1.0, 'born_after_1980': 1.0},
        'Elvis Presley': {'is_male': 1.0, 'is_real_person': 1.0, 'is_american': 1.0, 'is_singer': 1.0, 'died_before_2000': 1.0, 'has_dark_hair': 1.0, 'won_grammy': 1.0},
        'Michael Jackson': {'is_male': 1.0, 'is_real_person': 1.0, 'is_american': 1.0, 'is_singer': 1.0, 'has_dark_hair': 1.0, 'won_grammy': 1.0},

        # Historical & Scientific Figures
        'Albert Einstein': {'is_male': 1.0, 'is_real_person': 1.0, 'is_european': 1.0, 'is_scientist': 1.0, 'wears_glasses': 1.0, 'has_facial_hair': 1.0},
        'Marie Curie': {'is_female': 1.0, 'is_real_person': 1.0, 'is_european': 1.0, 'is_scientist': 1.0, 'died_before_2000': 1.0, 'has_dark_hair': 1.0},
        'William Shakespeare': {'is_male': 1.0, 'is_real_person': 1.0, 'is_european': 1.0, 'is_writer': 1.0, 'died_before_2000': 1.0, 'has_facial_hair': 1.0},
        'Queen Elizabeth II': {'is_female': 1.0, 'is_real_person': 1.0, 'is_european': 1.0, 'is_politician': 1.0},
        'Martin Luther King Jr.': {'is_male': 1.0, 'is_real_person': 1.0, 'is_american': 1.0, 'is_politician': 1.0, 'died_before_2000': 1.0, 'has_dark_hair': 1.0},

        # Athletes
        'Michael Jordan': {'is_male': 1.0, 'is_real_person': 1.0, 'is_american': 1.0, 'is_athlete': 1.0, 'is_alive': 1.0},
        'Serena Williams': {'is_female': 1.0, 'is_real_person': 1.0, 'is_american': 1.0, 'is_athlete': 1.0, 'is_alive': 1.0, 'has_dark_hair': 1.0, 'born_after_1980': 1.0},
        'Cristiano Ronaldo': {'is_male': 1.0, 'is_real_person': 1.0, 'is_european': 1.0, 'is_athlete': 1.0, 'is_alive': 1.0, 'has_dark_hair': 1.0, 'born_after_1980': 1.0},
        'Lionel Messi': {'is_male': 1.0, 'is_real_person': 1.0, 'is_athlete': 1.0, 'is_alive': 1.0, 'has_dark_hair': 1.0, 'born_after_1980': 1.0},

        # Fictional Characters (Live Action)
        'Spider-Man': {'is_male': 1.0, 'is_fictional': 1.0, 'is_american': 1.0, 'has_superpowers': 1.0, 'starred_in_marvel_movie': 1.0, 'is_humanoid': 1.0, 'known_for_comedy': 1.0},
        'Wonder Woman': {'is_female': 1.0, 'is_fictional': 1.0, 'has_superpowers': 1.0, 'starred_in_dc_movie': 1.0, 'is_humanoid': 1.0, 'has_dark_hair': 1.0},
        'Iron Man': {'is_male': 1.0, 'is_fictional': 1.0, 'is_american': 1.0, 'has_superpowers': 1.0, 'starred_in_marvel_movie': 1.0, 'is_humanoid': 1.0, 'has_dark_hair': 1.0, 'has_facial_hair': 1.0},
        'Batman': {'is_male': 1.0, 'is_fictional': 1.0, 'is_american': 1.0, 'starred_in_dc_movie': 1.0, 'is_humanoid': 1.0, 'has_dark_hair': 1.0},
        'Harry Potter': {'is_male': 1.0, 'is_fictional': 1.0, 'is_european': 1.0, 'from_usa_movie': 1.0, 'is_humanoid': 1.0, 'has_dark_hair': 1.0, 'wears_glasses': 1.0, 'has_superpowers': 1.0},
        'Katniss Everdeen': {'is_female': 1.0, 'is_fictional': 1.0, 'is_american': 1.0, 'from_usa_movie': 1.0, 'is_humanoid': 1.0, 'has_dark_hair': 1.0, 'is_athlete': 1.0},
        'Darth Vader': {'is_male': 1.0, 'is_fictional': 1.0, 'from_usa_movie': 1.0, 'is_humanoid': 1.0, 'has_superpowers': 1.0},
        'Yoda': {'is_male': 1.0, 'is_fictional': 1.0, 'from_usa_movie': 1.0, 'is_humanoid': 1.0, 'has_superpowers': 1.0},

        # Fictional Characters (Animated)
        'Pikachu': {'is_fictional': 1.0, 'is_animal': 1.0, 'is_asian': 1.0, 'has_superpowers': 1.0, 'from_tv_show': 1.0},
        'Mickey Mouse': {'is_male': 1.0, 'is_fictional': 1.0, 'is_american': 1.0, 'is_animal': 1.0, 'from_tv_show': 1.0, 'known_for_comedy': 1.0},
        'Homer Simpson': {'is_male': 1.0, 'is_fictional': 1.0, 'is_american': 1.0, 'from_tv_show': 1.0, 'is_humanoid': 1.0, 'known_for_comedy': 1.0},
        'Elsa': {'is_female': 1.0, 'is_fictional': 1.0, 'from_usa_movie': 1.0, 'is_humanoid': 1.0, 'has_blonde_hair': 1.0, 'has_superpowers': 1.0, 'is_singer': 1.0},
        'Shrek': {'is_male': 1.0, 'is_fictional': 1.0, 'from_usa_movie': 1.0, 'is_humanoid': 1.0, 'known_for_comedy': 1.0},
    }

    # Get the master list of all possible attribute IDs from the questions file
    attribute_ids = []
    with open(QUESTIONS_FILE_PATH, 'r') as f:
        next(f)  # Skip the header line
        for line in f:
            line = line.strip()
            if line:
                attribute_ids.append(line.split('::')[0])

    # Build the list of records for the DataFrame
    records = []
    for name, answered_yes_attrs in characters.items():
        record = {'CelebrityName': name}
        for attr_id in attribute_ids:
            # If the attribute is in the character's dictionary, it's a 1.0 (Yes).
            # Otherwise, it's a 0.0 (No). This ensures no missing values.
            if attr_id in answered_yes_attrs:
                record[attr_id] = 1.0
            else:
                record[attr_id] = 0.0
        records.append(record)
    
    # Create the final DataFrame from the records
    df = pd.DataFrame(records)

    # Ensure all attribute columns are of type float
    for col in attribute_ids:
        df[col] = df[col].astype(float)

    # Save the complete and consistent dataset to a parquet file
    df.to_parquet(CELEBRITIES_FILE_PATH, index=False, engine='pyarrow')
    print(f"Generated {CELEBRITIES_FILE_PATH} with {len(df)} celebrities.")
    print("Dataset is now complete: every character has a 'Yes' (1.0) or 'No' (0.0) for all questions.")

if __name__ == "__main__":
    generate_questions_file()
    generate_celebrities_parquet()
    print("\n--------------------------------------------------")
    print("Sample data generation complete.")
    print("Next Step: Run 'python train_model.py' to build the decision tree model from this new data.")
    print("--------------------------------------------------")