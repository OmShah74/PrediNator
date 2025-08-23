from predinator_core.game_engine import GameEngine
from predinator_core.learning_module import LearningModule
import time

class AkinatorService:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(AkinatorService, cls).__new__(cls, *args, **kwargs)
            print(f"[{time.ctime()}] AkinatorService: Creating new instance.")
        else:
            print(f"[{time.ctime()}] AkinatorService: Returning existing instance.")
        return cls._instance

    def __init__(self):
        # Ensure __init__ logic runs only once for the singleton
        if hasattr(self, '_initialized') and self._initialized:
            return
        
        print(f"[{time.ctime()}] AkinatorService __init__: Initializing GameEngine and LearningModule...")
        try:
            self.game_engine = GameEngine() # This will load/train model on init
            if not self.game_engine.tree_handler.model:
                 print(f"[{time.ctime()}] AkinatorService __init__: WARNING - Game model failed to load/train in GameEngine.")
            else:
                 print(f"[{time.ctime()}] AkinatorService __init__: GameEngine model seems loaded/trained.")

            self.learning_module = LearningModule(self.game_engine.tree_handler) # Pass the engine's tree_handler
            print(f"[{time.ctime()}] AkinatorService __init__: Initialization complete.")
            self._initialized = True
        except Exception as e:
            print(f"[{time.ctime()}] AkinatorService __init__: CRITICAL ERROR during initialization: {e}")
            import traceback
            traceback.print_exc()
            # Set them to None so checks in views can fail gracefully
            self.game_engine = None
            self.learning_module = None
            self._initialized = False # Mark as not successfully initialized


    def get_engine(self):
        if not self._initialized or not self.game_engine or not self.game_engine.tree_handler.model:
            print(f"[{time.ctime()}] AkinatorService.get_engine(): Service or engine/model not properly initialized. Attempting re-init...")
            # Potentially re-run __init__ logic here carefully or raise an error
            # For now, just return current state. Views should handle None engine.
            if not self._initialized : self.__init__() # Try to re-init if it never finished

        return self.game_engine

    def get_learner(self):
        if not self._initialized or not self.learning_module:
            print(f"[{time.ctime()}] AkinatorService.get_learner(): Service or learner not properly initialized.")
            if not self._initialized : self.__init__()
        return self.learning_module

akinator_service = AkinatorService() # Global instance

def get_global_game_engine():
    service_engine = akinator_service.get_engine()
    if service_engine is None:
        print(f"[{time.ctime()}] get_global_game_engine: CRITICAL - AkinatorService returned a None engine.")
    return service_engine

def get_global_learning_module():
    service_learner = akinator_service.get_learner()
    if service_learner is None:
        print(f"[{time.ctime()}] get_global_learning_module: CRITICAL - AkinatorService returned a None learner.")
    return service_learner