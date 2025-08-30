# predinator/train_model.py
import os
import django

# Set up Django environment to use the project's components
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'predinator_config.settings')
django.setup()

from predinator_core.tree_builder import AkinatorTree
from predinator_core.data_manager import load_celebrity_data, load_questions
from predinator_core.utils import CELEBRITIES_FILE, QUESTIONS_FILE
import time

def main():
    """
    A dedicated script to train the Akinator model.
    """
    print(f"[{time.ctime()}] --- Starting Model Training ---")

    # 1. Load data
    print(f"[{time.ctime()}] Loading data from {CELEBRITIES_FILE} and {QUESTIONS_FILE}...")
    celebrities_df = load_celebrity_data()
    questions_list = load_questions()

    if celebrities_df.empty or not questions_list:
        print(f"[{time.ctime()}] ERROR: Cannot train. Data is missing or empty. Please run generate_sample_data.py first.")
        return

    print(f"[{time.ctime()}] Data loaded. Found {len(celebrities_df)} celebrities and {len(questions_list)} questions.")

    # 2. Initialize the tree builder
    # These parameters are good for starting. As your data grows, you might tune ccp_alpha.
    tree_handler = AkinatorTree(
        ccp_alpha=0.0,
        min_samples_leaf=1,
        min_samples_split=2
    )

    # 3. Train the model
    print(f"[{time.ctime()}] Starting training process...")
    success = tree_handler.train(celebrities_df, questions_list)

    if success:
        print(f"[{time.ctime()}] --- Model Training Successful ---")
        print("Model and metadata have been saved to the 'data/model' directory.")
    else:
        print(f"[{time.ctime()}] --- Model Training FAILED ---")
        print("Please check the logs above for errors related to data preparation or model fitting.")

if __name__ == "__main__":
    main()