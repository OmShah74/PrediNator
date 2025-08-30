# PREDINATOR/game_app/utils_view_helpers.py
import numpy as np
import pandas as pd
import time

def get_session_game_state(request_session, game_engine_instance):
    """
    Loads game state from the Django session into the provided game_engine_instance.
    This function is now refined to ONLY reset the game if the model has changed
    or if it's a brand new session, NOT just because a game ended.
    """
    print(f"[{time.ctime()}] HELPER: get_session_game_state - START.")

    if not game_engine_instance or not game_engine_instance.tree_handler.model:
        print(f"[{time.ctime()}] HELPER Error: Game engine or model not available.")
        # Set session state to force an error/feedback page gracefully.
        request_session['akinator_game_active'] = False
        request_session['akinator_feedback_mode'] = True
        return

    current_model_id = id(game_engine_instance.tree_handler.model)
    session_model_id = request_session.get('akinator_model_id')
    
    reset_needed = False
    reset_reason = ""

    # Scenarios that trigger a full game reset:
    if 'akinator_current_node_id' not in request_session:
        reset_needed = True
        reset_reason = "new session or essential key missing"
    elif session_model_id != current_model_id:
        reset_needed = True
        reset_reason = f"model has been retrained (session: {session_model_id}, current: {current_model_id})"

    if reset_needed:
        print(f"[{time.ctime()}] HELPER Session: Resetting game state because: {reset_reason}.")
        
        # Start a new game within the engine instance
        if not game_engine_instance.start_new_game():
            print(f"[{time.ctime()}] HELPER Error: game_engine.start_new_game() FAILED.")
            request_session['akinator_game_active'] = False
            request_session['akinator_feedback_mode'] = True
            return

        # Copy the fresh state from the engine into the session
        request_session['akinator_current_node_id'] = int(game_engine_instance.current_node_id)
        request_session['akinator_path_taken'] = []
        request_session['akinator_game_active'] = True
        request_session['akinator_model_id'] = current_model_id
        request_session['akinator_last_guess'] = None
        request_session['akinator_feedback_mode'] = False
        print(f"[{time.ctime()}] HELPER Session: New game state initialized in session.")
    else:
        # If no reset is needed, simply load the existing state from session into the engine.
        # This will correctly load the state of a finished game (game_active=False) when needed.
        game_engine_instance.current_node_id = request_session.get('akinator_current_node_id', 0)
        game_engine_instance.path_taken = request_session.get('akinator_path_taken', [])
        game_engine_instance.game_active = request_session.get('akinator_game_active', True)
        print(f"[{time.ctime()}] HELPER Session: Loaded existing state into engine. Active: {game_engine_instance.game_active}")

    request_session.modified = True
    print(f"[{time.ctime()}] HELPER: get_session_game_state - END.")
    return


def update_session_game_state(request_session, game_engine_instance):
    """
    Saves the current state of the game_engine_instance to the Django session.
    (No changes needed here, but kept for completeness).
    """
    if not game_engine_instance:
        print(f"[{time.ctime()}] HELPER Error: Game engine not available in update_session_game_state.")
        return

    request_session['akinator_current_node_id'] = int(game_engine_instance.current_node_id)
    
    # The logic for serializing the path is complex, so we ensure it remains correct.
    serializable_path_taken = []
    if isinstance(game_engine_instance.path_taken, list):
        for item in game_engine_instance.path_taken:
            if isinstance(item, dict):
                serializable_item = {
                    key: int(value) if isinstance(value, np.integer) else
                         (None if pd.isna(value) else float(value)) if isinstance(value, np.floating) else
                         value
                    for key, value in item.items()
                }
                serializable_path_taken.append(serializable_item)
    
    request_session['akinator_path_taken'] = serializable_path_taken
    request_session['akinator_game_active'] = bool(game_engine_instance.game_active)
    
    request_session.modified = True
    print(f"[{time.ctime()}] HELPER Session: Updated state saved. Node: {request_session['akinator_current_node_id']}, Active: {request_session['akinator_game_active']}")