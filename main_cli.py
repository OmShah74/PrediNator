from predinator_core.game_engine import GameEngine
from predinator_core.learning_module import LearningModule
from predinator_core.data_manager import load_celebrity_data # For checking if celeb exists
import pandas as pd
from predinator_core.utils import CELEBRITIES_FILE, QUESTIONS_FILE              


def play_game():
    engine = GameEngine()
    # Pass the same tree_handler instance to the learner
    learner = LearningModule(engine.tree_handler)

    while True: # Main game loop for multiple plays
        if not engine.start_new_game():
            print("Exiting game due to initialization failure.")
            return

        guessed_celebrity = None
        while engine.game_active: # Inner loop for current game session
            question_obj, is_leaf = engine.get_next_question()

            if is_leaf:
                guessed_celebrity = engine.make_guess()
                break # Exit inner loop to handle guess outcome

            if question_obj:
                while True: # Loop for getting a valid answer
                    # Show possible answers from question_obj for clarity
                    ans_options_display = "/".join(ans.capitalize() for ans in question_obj.possible_answers)
                    ans = input(f"Q: {question_obj.text} ({ans_options_display}): ").strip().lower()
                    
                    # Allow shortcuts
                    if ans == 'y': ans = 'yes'
                    elif ans == 'n': ans = 'no'
                    elif ans == 'd' or ans == 'dk': ans = "don't know"

                    if engine.process_answer(ans):
                        break # Valid answer processed
                    # If process_answer returns False, it already printed an error message
            else: # Should ideally not happen if is_leaf is false
                print("Error: No question returned, but not at a leaf node. Ending current game.")
                engine.game_active = False # End this game session

        # --- After guess or if game ended prematurely ---
        if guessed_celebrity:
            correct_ans = input("Was my guess correct? (yes/no): ").strip().lower()
            if correct_ans.startswith('y'):
                print("Awesome! I win! 🎉")
            else:
                handle_incorrect_guess(learner, guessed_celebrity, engine.path_taken)
        elif not engine.game_active and not guessed_celebrity : # Game ended without a guess
            print("\nI couldn't make a guess with the information available.")
            print("This could be because the character is new or very unique based on current questions.")
            add_new_celeb_prompt(learner, engine.path_taken)


        play_again = input("\nDo you want to play again? (yes/no): ").strip().lower()
        if not play_again.startswith('y'):
            print("Thanks for playing! Goodbye!")
            break # Exit main game loop

def add_new_celeb_prompt(learner, game_path, guessed_celebrity_name=None):
    """Handles prompting to add a new celebrity when no guess was made or guess was wrong."""
    add_choice = input("Would you like to add the character you were thinking of? (yes/no): ").strip().lower()
    if add_choice.startswith('y'):
        actual_celebrity_name = input("Who were you thinking of? ").strip()
        if not actual_celebrity_name:
            print("No name provided. Cannot learn.")
            return

        df_celebs = load_celebrity_data() # Load fresh data to check
        if actual_celebrity_name in df_celebs['CelebrityName'].values:
            print(f"'{actual_celebrity_name}' is already in the database.")
            # Optional: "My apologies for not guessing correctly. Would you like to update its attributes?"
        else:
            if guessed_celebrity_name: # This path is usually for incorrect guess scenario
                 learner.add_new_question_and_learn(guessed_celebrity_name, actual_celebrity_name, game_path)
            else: # No guess was made, just learn the new celebrity
                 learner.learn_new_celebrity(actual_celebrity_name, game_path)
    else:
        print("Okay, maybe next time!")


def handle_incorrect_guess(learner: LearningModule, guessed_celebrity: str, game_path: list):
    actual_celebrity_name = input("Oh no! Who were you thinking of? ").strip()
    if not actual_celebrity_name:
        print("No name provided. Cannot learn.")
        return

    df_celebs = load_celebrity_data() # Load fresh data
    if actual_celebrity_name in df_celebs['CelebrityName'].values:
        print(f"It seems '{actual_celebrity_name}' is already in my database. My apologies, I should have known!")
        # TODO: Offer to refine attributes for existing celebrity or troubleshoot why it wasn't guessed.
    else:
        print(f"Okay, I don't know '{actual_celebrity_name}'. Let's learn about them.")
        add_q_choice = input(f"Would you like to add a new question to help distinguish '{actual_celebrity_name}' from my guess '{guessed_celebrity}' (and others)? (yes/no): ").strip().lower()
        if add_q_choice.startswith('y'):
            learner.add_new_question_and_learn(guessed_celebrity, actual_celebrity_name, game_path)
        else:
            print(f"Okay, I'll just try to learn about '{actual_celebrity_name}' using the current set of questions.")
            learner.learn_new_celebrity(actual_celebrity_name, game_path)

if __name__ == "__main__":
    print("Welcome to CLI Akinator - Advanced Edition!")
    print("=" * 40)
    print("Initializing...")
    # Initial check if data files exist, guide user if not.
    # The GameEngine's start_new_game() will also handle this.
    try:
        pd.read_parquet(CELEBRITIES_FILE) # Quick check
        open(QUESTIONS_FILE, 'r').close()
    except FileNotFoundError:
        print("\nIMPORTANT: Data files (celebrities.parquet or questions.txt) not found.")
        print("Please run 'python generate_sample_data.py' first to create sample files.")
        print("After running, you'll need to significantly expand 'data/celebrities.parquet' for good results.")
        exit()
    
    play_game()