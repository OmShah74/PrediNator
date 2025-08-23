# PREDINATOR/game_app/views.py
from django.shortcuts import render, redirect
from django.contrib import messages
import json
import pandas as pd
import time

from .game_services import get_global_game_engine, get_global_learning_module
from .utils_view_helpers import get_session_game_state, update_session_game_state
from predinator_core.data_manager import load_celebrity_data
from predinator_core.utils import DONT_KNOW_NUMERIC # Not directly used but good to be aware of

# --- Main Game Views ---
def play_view(request):
    print(f"[{time.ctime()}] VIEWS: play_view - Top. Method: {request.method}")
    game_engine = get_global_game_engine()
    if not game_engine or not game_engine.tree_handler or not game_engine.tree_handler.model: # More robust check
        print(f"[{time.ctime()}] VIEWS: play_view - Engine or model missing! Rendering error page.")
        messages.error(request, "Akinator model is not available. Please check server logs or data files.")
        return render(request, 'game_app/error.html', {'message': 'Akinator model is unavailable.'})

    # get_session_game_state will sync the global game_engine instance with this user's session data
    # or reset it if needed (e.g., new session, model change, game was inactive).
    get_session_game_state(request.session, game_engine)

    print(f"[{time.ctime()}] VIEWS: play_view - Post get_session_game_state. Session Active: {request.session.get('akinator_game_active')}, FeedbackMode: {request.session.get('akinator_feedback_mode')}")
    print(f"[{time.ctime()}] VIEWS: play_view - Engine state: Active: {game_engine.game_active}, Node: {game_engine.current_node_id}")


    # Rely on session state for redirection control, which should reflect engine state after get_session_game_state
    if not request.session.get('akinator_game_active', False) or \
       request.session.get('akinator_feedback_mode', False):
        print(f"[{time.ctime()}] VIEWS: play_view - Redirecting to learn_feedback (based on session).")
        return redirect('game_app:learn_feedback')

    # Now, use the global engine (which has been synced) to get the next question
    question_obj, is_leaf = game_engine.get_next_question() # This uses engine's current_node_id

    if is_leaf or not question_obj:
        print(f"[{time.ctime()}] VIEWS: play_view - Is leaf or no question. Time to guess. Engine Node: {game_engine.current_node_id}")
        guessed_celebrity = game_engine.make_guess() # Engine's game_active becomes False here
        
        request.session['akinator_last_guess'] = guessed_celebrity
        # Update session based on engine's state AFTER make_guess
        request.session['akinator_game_active'] = game_engine.game_active # Should be False
        request.session['akinator_feedback_mode'] = True
        
        # Persist the engine's final node (leaf node) and new game_active state to session
        update_session_game_state(request.session, game_engine)
        print(f"[{time.ctime()}] VIEWS: play_view - Guess made ('{guessed_celebrity}'), session updated, redirecting to learn_feedback.")
        return redirect('game_app:learn_feedback')

    print(f"[{time.ctime()}] VIEWS: play_view - Rendering question: '{question_obj.text}' for engine node {game_engine.current_node_id}")
    context = {
        'question_text': question_obj.text,
        'possible_answers': question_obj.possible_answers,
    }
    return render(request, 'game_app/play.html', context)


def answer_view(request):
    print(f"[{time.ctime()}] VIEWS: answer_view called. Method: {request.method}")
    game_engine = get_global_game_engine()
    if not game_engine:
        messages.error(request, "Game engine is not available.")
        return redirect('game_app:play')

    if request.method == 'POST':
        get_session_game_state(request.session, game_engine) # Sync global engine with this user's session

        print(f"[{time.ctime()}] VIEWS: answer_view - Post get_session_game_state. Session Active: {request.session.get('akinator_game_active')}, Engine Active: {game_engine.game_active}")

        if not request.session.get('akinator_game_active', True): # Check game status FROM SESSION
            print(f"[{time.ctime()}] VIEWS: answer_view - Game already ended in session, redirecting to learn_feedback.")
            messages.info(request, "The game has already ended or is in feedback mode.")
            return redirect('game_app:learn_feedback')

        user_answer_str = request.POST.get('answer')
        print(f"[{time.ctime()}] VIEWS: answer_view - User answer from POST: '{user_answer_str}' for engine node {game_engine.current_node_id}")
        
        if user_answer_str:
            # process_answer updates the engine's current_node_id and path_taken
            processed_ok = game_engine.process_answer(user_answer_str)
            if processed_ok:
                print(f"[{time.ctime()}] VIEWS: answer_view - Answer processed by engine. Engine new node: {game_engine.current_node_id}")
                update_session_game_state(request.session, game_engine) # Save engine's new state to session
                print(f"[{time.ctime()}] VIEWS: answer_view - Session updated. Session new node: {request.session.get('akinator_current_node_id')}")
            else:
                # This case means engine.process_answer itself returned False (e.g. invalid answer, or was already at leaf)
                print(f"[{time.ctime()}] VIEWS: answer_view - Engine.process_answer returned False.")
                messages.warning(request, "Could not process that answer. The game might have already reached a conclusion or the answer was invalid.")
                # If process_answer made engine.game_active false, ensure session reflects it
                if not game_engine.game_active:
                    print(f"[{time.ctime()}] VIEWS: answer_view - Engine became inactive after process_answer fail. Updating session.")
                    update_session_game_state(request.session, game_engine) # Persist inactive state
                return redirect('game_app:play') # Let play_view re-evaluate state (likely redirect to learn_feedback)
        else:
            print(f"[{time.ctime()}] VIEWS: answer_view - No answer found in POST data.")
            messages.warning(request, "No answer was submitted.")

        return redirect('game_app:play') # Go to next question or guess page (play_view will handle it)
    
    print(f"[{time.ctime()}] VIEWS: answer_view - Not a POST request, redirecting to play.")
    return redirect('game_app:play')

# --- Learning and Feedback Views ---
def learn_feedback_view(request):
    print(f"[{time.ctime()}] VIEWS: learn_feedback_view - Top.")
    game_engine = get_global_game_engine()
    if not game_engine:
         messages.error(request, "Game engine is not available.")
         return render(request, 'game_app/error.html', {'message': 'Game engine unavailable.'})
    
    get_session_game_state(request.session, game_engine) # Sync session, though game should be inactive

    last_guess = request.session.get('akinator_last_guess')
    session_game_active = request.session.get('akinator_game_active', False)
    session_feedback_mode = request.session.get('akinator_feedback_mode', False)

    print(f"[{time.ctime()}] VIEWS: learn_feedback_view - "
          f"LastGuess: {last_guess}, SessionActive: {session_game_active}, SessionFeedback: {session_feedback_mode}")

    game_is_over_for_template = session_feedback_mode or not session_game_active

    context = {
        'last_guess': last_guess,
        'game_ended_no_guess': game_is_over_for_template and not last_guess,
        'is_feedback_mode_active': session_feedback_mode,
        'is_game_session_active': session_game_active
    }
    return render(request, 'game_app/learn_feedback.html', context)

def process_learning_view(request):
    print(f"[{time.ctime()}] VIEWS: process_learning_view - Top. Method: {request.method}")
    learning_module = get_global_learning_module()
    game_engine = get_global_game_engine() # Needed to pass to get_session_game_state
    if not learning_module or not game_engine:
        messages.error(request, "Learning service or game engine is unavailable.")
        return render(request, 'game_app/error.html', {'message': 'Learning service unavailable.'})

    if request.method == 'POST':
        # Sync session state just in case, though mostly for path_taken
        get_session_game_state(request.session, game_engine)
        
        path_taken_from_session = request.session.get('akinator_path_taken', [])
        last_guess_from_session = request.session.get('akinator_last_guess')
        action = request.POST.get('action')
        actual_celebrity_name = request.POST.get('actual_celebrity_name', '').strip()
        print(f"[{time.ctime()}] VIEWS: process_learning_view - Action: '{action}', CelebName: '{actual_celebrity_name}'")


        if action == 'correct_guess':
            messages.success(request, "Great! I knew it!")
            request.session['akinator_feedback_mode'] = False # Exit feedback mode
            # Game is already inactive, reset_game will handle full reset for next game
            return redirect('game_app:reset_game')

        elif (action == 'incorrect_guess' and actual_celebrity_name) or \
             (action == 'no_guess_learn' and actual_celebrity_name):
            
            df_celebs = load_celebrity_data()
            if action == 'incorrect_guess' and actual_celebrity_name in df_celebs['CelebrityName'].values:
                messages.info(request, f"'{actual_celebrity_name}' is already in my database. My apologies for the wrong guess!")
                request.session['akinator_feedback_mode'] = False
                return redirect('game_app:reset_game')

            add_new_q_choice = request.POST.get('add_new_question_option')
            if action == 'incorrect_guess' and add_new_q_choice == 'yes':
                print(f"[{time.ctime()}] VIEWS: process_learning_view - Redirecting to add_question_form.")
                request.session['learn_info_for_new_question'] = {
                    'guessed_celebrity': last_guess_from_session,
                    'actual_celebrity': actual_celebrity_name,
                    'game_path': path_taken_from_session
                }
                return redirect('game_app:add_question_form')
            else: # Just learn celebrity (incorrect_guess with 'no' OR no_guess_learn)
                print(f"[{time.ctime()}] VIEWS: process_learning_view - Redirecting to learn_attributes.")
                game_path_answers = {item['attribute_id']: item['answer'] for item in path_taken_from_session}
                all_system_questions = learning_module.all_questions_list # Assumes it's up-to-date
                if not all_system_questions: # Safety check
                    messages.error(request, "No questions available in the system to learn attributes.")
                    return redirect('game_app:learn_feedback')
                
                questions_to_ask_on_form = []
                for q_obj_master in all_system_questions:
                    current_answer = game_path_answers.get(q_obj_master.attribute_id)
                    questions_to_ask_on_form.append({
                        'attribute_id': q_obj_master.attribute_id,
                        'text': q_obj_master.text,
                        'current_answer': current_answer, # For pre-filling the form
                        'current_answer_is_nan': pd.isna(current_answer) if current_answer is not None else True
                    })
                
                request.session['context_for_attribute_form'] = {
                    'celebrity_name': actual_celebrity_name,
                    'questions_to_ask': questions_to_ask_on_form,
                    'game_path_json': json.dumps(game_path_answers), # Original path that led here
                    'form_action': 'submit_new_celebrity_attributes' # Action for the attribute collection form
                }
                return redirect('game_app:learn_attributes')

        elif action == 'submit_new_celebrity_attributes':
            print(f"[{time.ctime()}] VIEWS: process_learning_view - Action: submit_new_celebrity_attributes.")
            # actual_celebrity_name is from a hidden field in the attribute collection form
            game_path_json = request.POST.get('game_path_json', '{}')
            game_path_answers_from_original_game = json.loads(game_path_json)

            all_submitted_attributes_from_form = {} # key: attr_id, value: string "yes"/"no"/"dontknow"
            for key, value in request.POST.items():
                if key.startswith('attr_'): # These are from the learn_new_celebrity_attributes.html form
                    attr_id = key.replace('attr_', '')
                    all_submitted_attributes_from_form[attr_id] = value
            
            success = learning_module.learn_new_celebrity_fully_web(
                actual_celebrity_name,
                game_path_answers_from_original_game, # Answers from the original game path
                all_submitted_attributes_from_form    # All attributes collected from the dedicated form
            )
            if success:
                messages.success(request, f"Successfully learned about '{actual_celebrity_name}'!")
                request.session['akinator_model_id'] = None # CRITICAL: Model has changed
            else:
                messages.error(request, f"Failed to learn about '{actual_celebrity_name}'. Please check server logs.")
            
            request.session['akinator_feedback_mode'] = False
            request.session.pop('context_for_attribute_form', None) # Clean up session
            return redirect('game_app:reset_game')
        
        messages.warning(request, "Could not process the learning request due to unrecognized action or missing data.")
        request.session['akinator_feedback_mode'] = False
        return redirect('game_app:reset_game')

    # If GET request to process_learning, just go back to feedback page
    return redirect('game_app:learn_feedback')


def learn_attributes_view(request):
    print(f"[{time.ctime()}] VIEWS: learn_attributes_view - Top.")
    # This view now just renders the form using context prepared and stored in session
    context_for_form = request.session.get('context_for_attribute_form')
    if not context_for_form:
        messages.error(request, "Cannot collect attributes: required information missing from session. Please start over.")
        return redirect('game_app:learn_feedback') # Or reset_game
    
    print(f"[{time.ctime()}] VIEWS: learn_attributes_view - Rendering learn_new_celebrity_attributes.html for {context_for_form.get('celebrity_name')}")
    return render(request, 'game_app/learn_new_celebrity_attributes.html', context_for_form)


def add_question_form_view(request):
    print(f"[{time.ctime()}] VIEWS: add_question_form_view - Top.")
    learn_info = request.session.get('learn_info_for_new_question')
    form_error = request.session.pop('form_error', None) # Get and clear any form error
    if not learn_info:
        messages.warning(request, "No learning information found to add a new question. Please go through the feedback steps again.")
        return redirect('game_app:learn_feedback')
    context = {
        'learn_info': learn_info,
        'form_error': form_error,
    }
    return render(request, 'game_app/add_question_form.html', context)

def submit_new_question_view(request):
    print(f"[{time.ctime()}] VIEWS: submit_new_question_view - Top. Method: {request.method}")
    learning_module = get_global_learning_module()
    if not learning_module:
        messages.error(request, "Learning service is unavailable.")
        return render(request, 'game_app/error.html', {'message': 'Learning service unavailable.'})

    if request.method == 'POST':
        learn_info = request.session.get('learn_info_for_new_question')
        if not learn_info:
            messages.error(request, "Session data for adding question expired or missing. Please try again.")
            return redirect('game_app:learn_feedback')

        new_q_text = request.POST.get('new_question_text', '').strip()
        new_q_attr_id = request.POST.get('new_question_id', '').strip().lower().replace(" ", "_")
        ans_for_actual_str = request.POST.get('answer_for_actual', '').strip()
        ans_for_guessed_str = request.POST.get('answer_for_guessed', '').strip()
        
        print(f"[{time.ctime()}] VIEWS: submit_new_question_view - POST data: Qtext='{new_q_text}', Qid='{new_q_attr_id}', AnsActual='{ans_for_actual_str}', AnsGuessed='{ans_for_guessed_str}'")


        # Basic validation
        if not all([new_q_text, new_q_attr_id, ans_for_actual_str]):
            request.session['form_error'] = "Question text, ID, and answer for the new character are required."
            request.session['learn_info_for_new_question'] = learn_info # Re-pass info for form redisplay
            return redirect('game_app:add_question_form')
        
        guessed_celebrity = learn_info.get('guessed_celebrity')
        if guessed_celebrity and not ans_for_guessed_str:
            request.session['form_error'] = f"Answer for '{guessed_celebrity}' for the new question is required."
            request.session['learn_info_for_new_question'] = learn_info
            return redirect('game_app:add_question_form')
        elif not guessed_celebrity and not ans_for_guessed_str: # No guess made originally
             ans_for_guessed_str = "dontknow" # Default for the "guessed" character (which is non-existent here)

        # Call the method in learning_module that prepares for full attribute collection after adding Q
        context_for_next_step = learning_module.web_add_question_and_learn_redirect(
             guessed_celebrity_name=guessed_celebrity,
             actual_celebrity_name=learn_info['actual_celebrity'],
             game_path=learn_info['game_path'],
             new_q_text=new_q_text,
             new_q_attr_id=new_q_attr_id,
             ans_for_actual_new_q_str=ans_for_actual_str,
             ans_for_guessed_new_q_str=ans_for_guessed_str
        )

        if context_for_next_step:
            request.session['context_for_attribute_form'] = context_for_next_step
            request.session.pop('learn_info_for_new_question', None) # Clear this specific session var
            messages.success(request, f"New question '{new_q_text}' processed. Now, please provide all attributes for '{learn_info['actual_celebrity']}'.")
            print(f"[{time.ctime()}] VIEWS: submit_new_question_view - Redirecting to learn_attributes for '{learn_info['actual_celebrity']}'.")
            return redirect('game_app:learn_attributes')
        else:
            # Error from web_add_question_and_learn_redirect (e.g., question ID exists, invalid input)
            request.session['form_error'] = "Failed to process the new question. The Question ID might already exist, or data was invalid. Please check and try again."
            request.session['learn_info_for_new_question'] = learn_info # Re-pass info for form redisplay
            return redirect('game_app:add_question_form')

    print(f"[{time.ctime()}] VIEWS: submit_new_question_view - Not a POST, redirecting to add_question_form.")
    return redirect('game_app:add_question_form')


# --- Utility View ---
def reset_game_view(request):
    print(f"[{time.ctime()}] VIEWS: reset_game_view - Top.")
    keys_to_delete = [
        'akinator_current_node_id', 'akinator_path_taken', 'akinator_game_active',
        'akinator_last_guess', 'akinator_feedback_mode', 'learn_info_for_new_question',
        'akinator_model_id', 'error_message', 'form_error', 'context_for_attribute_form'
    ]
    for key in keys_to_delete:
        if key in request.session:
            del request.session[key]
            print(f"[{time.ctime()}] VIEWS: reset_game_view - Deleted session key: {key}")
            
    messages.info(request, "Game has been reset. Let's play!")
    # get_session_game_state will be called in play_view to re-initialize a fresh game
    # based on the cleared session.
    return redirect('game_app:play')