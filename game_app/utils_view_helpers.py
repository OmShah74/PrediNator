# PREDINATOR/game_app/utils_view_helpers.py
import numpy as np
import pandas as pd
import time

def get_session_game_state(request_session, game_engine_instance):
    """
    Loads game state from the Django session into the provided game_engine_instance.
    If no valid game state in session or model has changed, it resets the game.
    Ensures the global game_engine_instance is synced with this user's session data.
    """
    print(f"[{time.ctime()}] HELPER: get_session_game_state - START. Session ID: {request_session.session_key}")

    if not game_engine_instance or \
       not game_engine_instance.tree_handler or \
       not game_engine_instance.tree_handler.model: # Check if model itself is loaded in tree_handler
        print(f"[{time.ctime()}] HELPER Error: Game engine, tree_handler, or its model not available in get_session_game_state.")
        request_session['akinator_game_active'] = False
        request_session['akinator_feedback_mode'] = True # Force to feedback if engine is broken
        # Set defaults for other keys to prevent KeyErrors in templates/views
        request_session.setdefault('akinator_current_node_id', 0)
        request_session.setdefault('akinator_path_taken', [])
        request_session.setdefault('akinator_model_id', None)
        request_session.setdefault('akinator_last_guess', None)
        request_session.modified = True
        return

    current_model_id = id(game_engine_instance.tree_handler.model)
    session_model_id = request_session.get('akinator_model_id')
    session_game_active = request_session.get('akinator_game_active', False)

    reset_needed = False
    reset_reason = ""

    if 'akinator_current_node_id' not in request_session: # First time, or session was cleared
        reset_needed = True
        reset_reason = "new session or akinator_current_node_id missing"
    elif not session_game_active: # If session explicitly says game is inactive (e.g., after a game ended)
        reset_needed = True
        reset_reason = "game was marked inactive in session"
    elif session_model_id != current_model_id: # Model has been retrained
        reset_needed = True
        reset_reason = "model changed"
    
    # Debug: print current session state before deciding on reset
    # print(f"[{time.ctime()}] HELPER PRE-RESET CHECK: Session keys: {list(request_session.keys())}")
    # print(f"[{time.ctime()}] HELPER PRE-RESET CHECK: Current Node (Sess): {request_session.get('akinator_current_node_id')}, Active (Sess): {session_game_active}, ModelID (Sess): {session_model_id}, Feedback(Sess): {request_session.get('akinator_feedback_mode')}")


    if reset_needed:
        print(f"[{time.ctime()}] HELPER Session: Resetting game state due to: {reset_reason}. "
              f"SessionModelID was: {session_model_id}, CurrentModelID is: {current_model_id}")
        
        # Call start_new_game on the global engine instance.
        # This re-initializes ITS internal state (current_node, path_taken, game_active).
        success = game_engine_instance.start_new_game()
        if not success:
            print(f"[{time.ctime()}] HELPER Error: game_engine.start_new_game() FAILED during reset.")
            request_session['akinator_game_active'] = False
            request_session['akinator_feedback_mode'] = True # Send to feedback to show error
            request_session.modified = True
            return

        # Now, copy the engine's fresh state into the session.
        request_session['akinator_current_node_id'] = int(game_engine_instance.current_node_id)
        request_session['akinator_path_taken'] = [] # Path starts empty for a new session game
        request_session['akinator_game_active'] = True # Set explicitly after successful start
        request_session['akinator_model_id'] = current_model_id
        request_session['akinator_last_guess'] = None
        request_session['akinator_feedback_mode'] = False
        print(f"[{time.ctime()}] HELPER Session: New game state initialized IN SESSION. Node: {request_session['akinator_current_node_id']}, Active: {request_session['akinator_game_active']}")
    else:
        # Sync global engine's state from the existing valid session for this request
        game_engine_instance.current_node_id = request_session.get('akinator_current_node_id', 0)
        
        loaded_path_taken_from_session = request_session.get('akinator_path_taken', [])
        engine_internal_path = [] # Path for the engine instance to use for this request
        if isinstance(loaded_path_taken_from_session, list):
            for item in loaded_path_taken_from_session:
                if isinstance(item, dict):
                    engine_item = {}
                    for key, value in item.items():
                        if key == 'answer' and value is None: # JSON 'null' from session becomes Python None
                            engine_item[key] = np.nan # Convert back if engine logic expects np.nan
                        else:
                            engine_item[key] = value
                    engine_internal_path.append(engine_item)
        game_engine_instance.path_taken = engine_internal_path # Engine uses this path
        game_engine_instance.game_active = request_session.get('akinator_game_active', True) # Sync active state
        # print(f"[{time.ctime()}] HELPER Session: Loaded existing game state into engine. Engine Node: {game_engine_instance.current_node_id}, Engine Path len: {len(game_engine_instance.path_taken)}, Engine Active: {game_engine_instance.game_active}")
    
    request_session.modified = True # Mark session as modified as a general precaution if unsure
    print(f"[{time.ctime()}] HELPER: get_session_game_state - END. Session Game Active: {request_session.get('akinator_game_active')}, Engine Game Active: {game_engine_instance.game_active}")
    return


def update_session_game_state(request_session, game_engine_instance):
    """
    Saves the current state of the game_engine_instance to the Django session
    with proper type conversions for JSON serialization.
    """
    print(f"[{time.ctime()}] HELPER: update_session_game_state - START. Engine Node: {game_engine_instance.current_node_id}, Engine Active: {game_engine_instance.game_active}")
    if not game_engine_instance:
        print(f"[{time.ctime()}] HELPER Error: Game engine not available in update_session_game_state.")
        return

    request_session['akinator_current_node_id'] = int(game_engine_instance.current_node_id)
    
    serializable_path_taken = []
    # The game_engine_instance.path_taken is what needs to be saved to session
    if isinstance(game_engine_instance.path_taken, list):
        for item in game_engine_instance.path_taken:
            if isinstance(item, dict):
                serializable_item = {}
                for key, value in item.items():
                    if isinstance(value, np.integer):
                        serializable_item[key] = int(value)
                    elif isinstance(value, np.floating): # This includes np.nan
                        if np.isnan(value): # Specifically np.nan
                            serializable_item[key] = None # Store as None for JSON
                        else:
                            serializable_item[key] = float(value)
                    else: # str, bool, Python int/float are fine
                        serializable_item[key] = value
                serializable_path_taken.append(serializable_item)
            else: # Should not happen with current path_taken structure
                print(f"[{time.ctime()}] HELPER Warning: Unexpected item type in engine's path_taken during save: {type(item)}")
                serializable_path_taken.append(item) 
    else:
        print(f"[{time.ctime()}] HELPER Warning: game_engine_instance.path_taken is not a list during save: {type(game_engine_instance.path_taken)}")

    request_session['akinator_path_taken'] = serializable_path_taken
    request_session['akinator_game_active'] = bool(game_engine_instance.game_active)
    
    # Note: 'akinator_last_guess' and 'akinator_feedback_mode' are usually set directly in views.
    # 'akinator_model_id' is set in get_session_game_state when a new model/game is established.

    request_session.modified = True # Crucial: tell Django the session has changed
    print(f"[{time.ctime()}] HELPER Session: Updated state saved. Node: {request_session['akinator_current_node_id']}, "
          f"Path len: {len(request_session['akinator_path_taken'])}, Active: {request_session['akinator_game_active']}")
    print(f"[{time.ctime()}] HELPER: update_session_game_state - END")