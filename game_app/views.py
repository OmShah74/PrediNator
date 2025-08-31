# PREDINATOR/game_app/views.py
from django.shortcuts import render, redirect
from django.contrib import messages
import json
import pandas as pd
import time

from .game_services import get_global_game_engine, get_global_learning_module
from .utils_view_helpers import get_session_game_state, update_session_game_state
from predinator_core.data_manager import load_celebrity_data

# --- Main Game Views ---

def play_view(request):
    print(f"[{time.ctime()}] VIEWS: play_view - Top.")
    game_engine = get_global_game_engine()
    if not game_engine or not game_engine.tree_handler.model:
        messages.error(request, "Akinator model is not available. Please run train_model.py")
        return render(request, 'game_app/error.html', {'message': 'Akinator model is unavailable.'})

    get_session_game_state(request.session, game_engine)

    if not request.session.get('akinator_game_active', False) or request.session.get('akinator_feedback_mode', False):
        return redirect('game_app:learn_feedback')

    question_obj, is_leaf = game_engine.get_next_question()

    if is_leaf or not question_obj:
        guessed_celebrity = game_engine.make_guess()
        print(f"[{time.ctime()}] VIEWS: play_view - Guess made: '{guessed_celebrity}'")
        
        request.session['akinator_last_guess'] = guessed_celebrity
        request.session['akinator_game_active'] = game_engine.game_active
        request.session['akinator_feedback_mode'] = True
        update_session_game_state(request.session, game_engine)
        
        return redirect('game_app:learn_feedback')

    context = {
        'question_text': question_obj.text,
        'possible_answers': question_obj.possible_answers,
    }
    return render(request, 'game_app/play.html', context)


def answer_view(request):
    if request.method != 'POST':
        return redirect('game_app:play')

    game_engine = get_global_game_engine()
    if not game_engine:
        messages.error(request, "Game engine is not available.")
        return redirect('game_app:play')

    get_session_game_state(request.session, game_engine)

    user_answer_str = request.POST.get('answer')
    if user_answer_str and game_engine.process_answer(user_answer_str):
        update_session_game_state(request.session, game_engine)
    else:
        messages.warning(request, "Could not process that answer.")

    return redirect('game_app:play')


# --- Learning and Feedback Views ---

def learn_feedback_view(request):
    print(f"[{time.ctime()}] VIEWS: learn_feedback_view - Top.")
    get_session_game_state(request.session, get_global_game_engine())

    last_guess = request.session.get('akinator_last_guess')
    game_active = request.session.get('akinator_game_active', False)

    context = {
        'last_guess': last_guess,
        'game_ended_no_guess': not game_active and not last_guess,
    }
    return render(request, 'game_app/learn_feedback.html', context)


def process_learning_view(request):
    if request.method != 'POST':
        return redirect('game_app:learn_feedback')

    learning_module = get_global_learning_module()
    game_engine = get_global_game_engine()
    if not learning_module or not game_engine:
        messages.error(request, "Learning service or game engine is unavailable.")
        return render(request, 'game_app/error.html', {'message': 'Learning service unavailable.'})

    get_session_game_state(request.session, game_engine)
    
    path_taken = request.session.get('akinator_path_taken', [])
    last_guess = request.session.get('akinator_last_guess')
    action = request.POST.get('action')
    actual_celebrity_name = request.POST.get('actual_celebrity_name', '').strip()

    if action == 'correct_guess':
        messages.success(request, "Great! I knew it!")
        return redirect('game_app:reset_game')

    elif action in ('incorrect_guess', 'no_guess_learn') and actual_celebrity_name:
        df_celebs = load_celebrity_data()
        if actual_celebrity_name in df_celebs['CelebrityName'].values:
            messages.info(request, f"'{actual_celebrity_name}' is already in my database. My apologies for the wrong guess!")
            return redirect('game_app:reset_game')

        if action == 'incorrect_guess' and request.POST.get('add_new_question_option') == 'yes':
            request.session['learn_info_for_new_question'] = {
                'guessed_celebrity': last_guess,
                'actual_celebrity': actual_celebrity_name,
                'game_path': path_taken
            }
            return redirect('game_app:add_question_form')
        else:
            game_path_answers = {item['attribute_id']: item['answer'] for item in path_taken}
            all_system_questions = learning_module.all_questions_list
            if not all_system_questions:
                messages.error(request, "No questions available in the system to learn attributes.")
                return redirect('game_app:learn_feedback')
            
            questions_to_ask = []
            for q_obj in all_system_questions:
                answer = game_path_answers.get(q_obj.attribute_id)
                questions_to_ask.append({
                    'attribute_id': q_obj.attribute_id,
                    'text': q_obj.text,
                    'current_answer': answer,
                    'current_answer_is_nan': pd.isna(answer) if answer is not None else True
                })
            
            request.session['context_for_attribute_form'] = {
                'celebrity_name': actual_celebrity_name,
                'questions_to_ask': questions_to_ask,
                'game_path_json': json.dumps(game_path_answers),
                'form_action': 'submit_new_celebrity_attributes'
            }
            return redirect('game_app:learn_attributes')

    elif action == 'submit_new_celebrity_attributes':
        all_submitted_attrs = {k.replace('attr_', ''): v for k, v in request.POST.items() if k.startswith('attr_')}
        
        success = learning_module.learn_new_celebrity_fully_web(
            actual_celebrity_name=actual_celebrity_name,
            game_path_answers=json.loads(request.POST.get('game_path_json', '{}')),
            all_submitted_attributes=all_submitted_attrs
        )
        
        # --- CRITICAL CHANGE FOR GRACEFUL FAILURE ---
        if success:
            messages.success(request, f"Successfully learned about '{actual_celebrity_name}' and retrained the model!")
            request.session['akinator_model_id'] = None # Invalidate model ID to force a fresh game state
        else:
            messages.error(request, f"Failed to learn about '{actual_celebrity_name}'. The existing model is still active. Please check server logs for details.")
            # Do NOT redirect to reset. Redirect to the main play page so the user can continue.
            request.session.pop('context_for_attribute_form', None)
            return redirect('game_app:play') # Go back to the game with the old model

        request.session.pop('context_for_attribute_form', None)
        return redirect('game_app:reset_game') # Only reset if learning was successful
    
    messages.warning(request, "Could not process the request.")
    return redirect('game_app:reset_game')


def learn_attributes_view(request):
    context = request.session.get('context_for_attribute_form')
    if not context:
        messages.error(request, "Cannot collect attributes: information missing from session. Please start over.")
        return redirect('game_app:learn_feedback')
    
    return render(request, 'game_app/learn_new_celebrity_attributes.html', context)


def add_question_form_view(request):
    learn_info = request.session.get('learn_info_for_new_question')
    if not learn_info:
        messages.warning(request, "No learning information found. Please start the feedback process again.")
        return redirect('game_app:learn_feedback')
    context = {
        'learn_info': learn_info,
        'form_error': request.session.pop('form_error', None),
    }
    return render(request, 'game_app/add_question_form.html', context)


def submit_new_question_view(request):
    if request.method != 'POST':
        return redirect('game_app:add_question_form')

    learning_module = get_global_learning_module()
    if not learning_module:
        messages.error(request, "Learning service is unavailable.")
        return render(request, 'game_app/error.html', {'message': 'Learning service unavailable.'})

    learn_info = request.session.get('learn_info_for_new_question')
    if not learn_info:
        messages.error(request, "Session data for adding a question has expired. Please try again.")
        return redirect('game_app:learn_feedback')

    new_q_text = request.POST.get('new_question_text', '').strip()
    new_q_attr_id = request.POST.get('new_question_id', '').strip().lower().replace(" ", "_")
    ans_for_actual = request.POST.get('answer_for_actual', '').strip()
    ans_for_guessed = request.POST.get('answer_for_guessed', '').strip()
    
    guessed_celebrity = learn_info.get('guessed_celebrity')

    if not all([new_q_text, new_q_attr_id, ans_for_actual]) or (guessed_celebrity and not ans_for_guessed):
        request.session['form_error'] = "All fields are required."
        return redirect('game_app:add_question_form')

    context_for_next_step = learning_module.web_add_question_and_learn_redirect(
         guessed_celebrity_name=guessed_celebrity,
         actual_celebrity_name=learn_info['actual_celebrity'],
         game_path=learn_info['game_path'],
         new_q_text=new_q_text,
         new_q_attr_id=new_q_attr_id,
         ans_for_actual_new_q_str=ans_for_actual,
         ans_for_guessed_new_q_str=ans_for_guessed or "dontknow"
    )

    if context_for_next_step:
        request.session['context_for_attribute_form'] = context_for_next_step
        request.session.pop('learn_info_for_new_question', None)
        messages.success(request, f"New question added. Now, please provide all attributes for '{learn_info['actual_celebrity']}'.")
        return redirect('game_app:learn_attributes')
    else:
        request.session['form_error'] = "Failed to process the new question. The Question ID might already exist."
        return redirect('game_app:add_question_form')


# --- Utility View ---

def reset_game_view(request):
    print(f"[{time.ctime()}] VIEWS: reset_game_view - Clearing all game-related session keys.")
    
    keys_to_clear = [
        'akinator_current_node_id', 
        'akinator_path_taken', 
        'akinator_game_active',
        'akinator_last_guess', 
        'akinator_feedback_mode', 
        'learn_info_for_new_question',
        'context_for_attribute_form',
        'form_error',
    ]
    
    for key in keys_to_clear:
        if key in request.session:
            del request.session[key]

    request.session['akinator_game_active'] = True
    request.session['akinator_feedback_mode'] = False

    messages.info(request, "Game has been reset. Let's play!")
    return redirect('game_app:play')