import pandas as pd
import numpy as np
import os
DATA_DIR = 'data'
QUESTIONS_FILE_PATH = os.path.join(DATA_DIR, 'questions.txt')
CELEBRITIES_FILE_PATH = os.path.join(DATA_DIR, 'celebrities.parquet')

def generate_questions_file():
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
    data = {
        'CelebrityName': [
            'Tom Hanks', 'Scarlett Johansson', 'Taylor Swift', 'Leonardo DiCaprio', 'Dwayne Johnson',
            'Elvis Presley', 'Meryl Streep', 'Will Smith', 'Albert Einstein', 'Marie Curie',
            'Serena Williams', 'Michael Jordan', 'Queen Elizabeth II', 'Donald Trump', 'Spider-Man',
            'Wonder Woman', 'Harry Potter', 'Mickey Mouse', 'Pikachu', 'William Shakespeare', 'Cleopatra',
            'Elon Musk', 'Oprah Winfrey', 'Keanu Reeves', 'Bill Gates' # Added a few more
        ]
    }
    df = pd.DataFrame(data)

    attribute_ids = []
    with open(QUESTIONS_FILE_PATH, 'r') as f:
        next(f) # Skip header
        for line in f:
            line = line.strip()
            if line:
                attr_id = line.split('::')[0]
                attribute_ids.append(attr_id)
                df[attr_id] = np.nan # Initialize all with DontKnow

    # --- Populate with SOME sample data (Yes=1.0, No=0.0, DontKnow=np.nan) ---
    # This is tedious and error-prone for many entries. A database or structured input source is better for large scale.
    
    # Tom Hanks
    df.loc[df['CelebrityName'] == 'Tom Hanks', ['is_male', 'is_real_person', 'is_american', 'is_actor', 'is_alive', 'won_oscar', 'has_dark_hair']] = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    df.loc[df['CelebrityName'] == 'Tom Hanks', ['is_female','is_fictional', 'is_singer', 'is_politician', 'has_superpowers', 'born_after_1980']] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    # Scarlett Johansson
    df.loc[df['CelebrityName'] == 'Scarlett Johansson', ['is_female', 'is_real_person', 'is_american', 'is_actor', 'is_alive', 'starred_in_marvel_movie', 'born_after_1980']] = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    df.loc[df['CelebrityName'] == 'Scarlett Johansson', ['is_male', 'is_fictional', 'won_oscar', 'has_blonde_hair']] = [0.0, 0.0, 0.0, 1.0] # Assuming typically blonde for roles

    # Taylor Swift
    df.loc[df['CelebrityName'] == 'Taylor Swift', ['is_female', 'is_real_person', 'is_american', 'is_singer', 'is_alive', 'won_grammy', 'born_after_1980', 'has_blonde_hair']] = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    df.loc[df['CelebrityName'] == 'Taylor Swift', ['is_male', 'is_fictional', 'is_actor']] = [0.0, 0.0, 0.0]

    # Albert Einstein
    df.loc[df['CelebrityName'] == 'Albert Einstein', ['is_male', 'is_real_person', 'is_european', 'is_scientist', 'died_before_2000', 'wears_glasses']] = [1.0, 1.0, 1.0, 1.0, 0.0, 1.0]
    df.loc[df['CelebrityName'] == 'Albert Einstein', ['is_fictional', 'is_alive', 'is_american', 'born_after_1980']] = [0.0, 0.0, 0.0, 0.0]

    # Spider-Man
    df.loc[df['CelebrityName'] == 'Spider-Man', ['is_male', 'is_fictional', 'is_american', 'has_superpowers', 'starred_in_marvel_movie', 'is_humanoid']] = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    df.loc[df['CelebrityName'] == 'Spider-Man', ['is_real_person', 'won_oscar', 'is_alive']] = [0.0, 0.0, np.nan] # Fictional "aliveness"

    # Keanu Reeves
    df.loc[df['CelebrityName'] == 'Keanu Reeves', ['is_male', 'is_real_person', 'is_actor', 'is_alive', 'has_dark_hair', 'starred_in_dc_movie']] = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0] # Constantine (DC-ish) / Matrix
    df.loc[df['CelebrityName'] == 'Keanu Reeves', ['is_female', 'is_fictional', 'is_american', 'won_oscar', 'born_after_1980']] = [0.0, 0.0, 0.0, 0.0, 0.0] # Born 1964, Canadian

    # Fill remaining specified characters attributes with 0.0 if not 1.0 or np.nan (for this small sample)
    # For a large dataset, meticulous data entry or sophisticated imputation is needed for NaNs.
    populated_celebs = ['Tom Hanks', 'Scarlett Johansson', 'Taylor Swift', 'Albert Einstein', 'Spider-Man', 'Keanu Reeves']
    for celeb_name in populated_celebs:
        for attr_id in attribute_ids:
            if pd.isna(df.loc[df['CelebrityName'] == celeb_name, attr_id].iloc[0]):
                df.loc[df['CelebrityName'] == celeb_name, attr_id] = 0.0 # Default to No if not specified Yes or Dk

    # Ensure all attribute columns are float to handle np.nan
    for col in attribute_ids:
        df[col] = df[col].astype(float)

    df.to_parquet(CELEBRITIES_FILE_PATH, index=False, engine='pyarrow')
    print(f"Generated sample {CELEBRITIES_FILE_PATH} with {len(df)} celebrities and {len(attribute_ids)} attributes.")
    print("IMPORTANT: This is a very small, illustrative sample. For the project's goal of ~1000 celebrities,")
    print("you MUST extensively populate 'celebrities.parquet' with accurate data for each character.")

if __name__ == "__main__":
    generate_questions_file()
    generate_celebrities_parquet()
    print("\nSample data generation complete.")
    print("Run 'python main_cli.py' to use the application.")