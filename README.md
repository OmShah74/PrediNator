# Predinator: The Django-Powered Akinator Game

Predinator is a web-based guessing game, similar to the classic Akinator, that uses a machine learning model to guess the celebrity or fictional character a user is thinking of. The application asks a series of "Yes/No/Don't Know" questions and uses a **Decision Tree Classifier** to narrow down the possibilities and make a final prediction.

The project is built with a decoupled architecture, separating the core machine learning logic (`predinator_core`) from the web interface (`game_app`), and includes a powerful learning module that allows the game to improve over time by learning new characters and questions directly from user feedback.

## ✨ Features

-   **Interactive Gameplay**: A clean, simple web interface for answering questions.
-   **Decision Tree Logic**: Utilizes a `scikit-learn` Decision Tree model for fast and interpretable predictions.
-   **Dynamic Learning**: The model can be retrained on the fly.
    -   **Learn New Characters**: If Predinator guesses incorrectly or cannot guess, users can teach it a new character by providing its attributes.
    -   **Add New Questions**: Users can add new, distinguishing questions to help the model differentiate between its incorrect guess and the user's actual character.
-   **Data Management**: Comes with scripts to generate a rich sample dataset and to train the model from scratch.
-   **Production Ready**: The project is configured for production deployment with Gunicorn, PostgreSQL, and Whitenoise for serving static files.
-   **CI/CD Pipeline**: Includes a pre-configured GitHub Actions workflow for Continuous Integration (testing) and Continuous Deployment to platforms like Render.

## 🚀 Local Development Setup

Follow these steps to get the Predinator application running on your local machine.

### Prerequisites

-   Python 3.10+
-   `pip` and `venv`

### Step-by-Step Installation

1.  **Clone the Repository**
    ```bash
    git clone <your-repository-url>
    cd predinator
    ```

2.  **Create and Activate a Virtual Environment**
    This is a crucial step to isolate the project's dependencies.
    ```bash
    # Create the virtual environment
    python -m venv venv

    # Activate it (Windows)
    .\venv\Scripts\activate

    # Activate it (macOS/Linux)
    source venv/bin/activate
    ```

3.  **Install Dependencies**
    The `requirements.txt` file contains all necessary packages for both development and production.
    ```bash
    pip install -r requirements.txt
    ```

4.  **Generate the Initial Dataset**
    This script creates the `questions.txt` and `celebrities.parquet` files inside the `data/` directory.
    ```bash
    python generate_sample_data.py
    ```

5.  **Train the Initial Model**
    This script reads the generated data and creates the `akinator_model.joblib` and `akinator_metadata.joblib` files inside the `data/model/` directory.
    ```bash
    python train_model.py
    ```

6.  **Run Django Migrations**
    This will create the local `db.sqlite3` database needed for Django's session management.
    ```bash
    python manage.py migrate
    ```

7.  **Run the Development Server**
    You're all set!
    ```bash
    python manage.py runserver
    ```
    The application will be available at `http://127.0.0.1:8000/`. The root URL automatically redirects to `/akinator/play/`.

## ⚙️ Deployment

This project is configured for a seamless CI/CD deployment to a PaaS provider like **Render**.

1.  **Configuration Files**:
    -   `requirements.txt`: Provides a frozen list of all dependencies for a stable build.
    -   `build.sh`: A simple shell script that tells Render how to build the project (install dependencies, collect static files, run migrations).
    -   `predinator_config/settings.py`: Automatically configures the database using the `DATABASE_URL` environment variable and serves static files using `whitenoise`.

2.  **CI/CD with GitHub Actions**:
    -   The workflow is defined in `.github/workflows/ci-cd.yml`.
    -   On every push to the `main` branch, the workflow automatically runs the Django tests.
    -   If the tests pass, it triggers a deployment on Render by sending a request to a secret "Deploy Hook" URL.

## 🛠️ Technology Stack

-   **Backend**: Django
-   **Machine Learning**: Scikit-learn
-   **Data Handling**: Pandas, NumPy, PyArrow
-   **Production Server**: Gunicorn
-   **Database**: PostgreSQL (Production), SQLite3 (Development)
-   **Static File Serving**: WhiteNoise
-   **CI/CD**: GitHub Actions
-   **Hosting**: Render

## 🔮 Future Improvements

-   **Asynchronous Training**: For a larger dataset, the retraining process can become slow. This could be moved to a background task queue (like Celery with Redis) to avoid tying up the web server.
-   **Stateless Game Engine**: Refactor the `GameEngine` to be stateless, passing the game state (current node, path taken) as arguments to its methods. This would improve concurrency and make the application more scalable.
-   **Enhanced UI/UX**: Improve the frontend with a modern JavaScript framework or better CSS to create a more dynamic user experience.
-   **User Accounts**: Add user registration to allow players to track their game history, stats, and contributions (characters they've taught the game).