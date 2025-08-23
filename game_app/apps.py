from django.apps import AppConfig
import time # For logging

class GameAppConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'game_app'

    def ready(self):
        # This method is called once Django starts up.
        # Import your service here to ensure it's initialized.
        try:
            # Check if running manage.py commands that should not initialize heavy services
            # For example, collectstatic, makemigrations, migrate
            import sys
            # A simple check, can be made more robust
            # Common commands that don't need the full app initialized
            avoid_init_commands = ['makemigrations', 'migrate', 'collectstatic', 'createsuperuser', 'check', 'shell']
            should_initialize = not any(cmd in sys.argv for cmd in avoid_init_commands)

            if should_initialize:
                print(f"[{time.ctime()}] GameAppConfig.ready(): Attempting to initialize game_services...")
                from . import game_services # Import your service module
                # Accessing the global instance ensures it's created
                _ = game_services.akinator_service # This triggers the __init__ of AkinatorService
                print(f"[{time.ctime()}] GameAppConfig.ready(): Akinator game services should be initialized.")
            else:
                print(f"[{time.ctime()}] GameAppConfig.ready(): Skipping full game_services initialization for command: {' '.join(sys.argv)}")

        except ImportError as e:
            print(f"[{time.ctime()}] GameAppConfig.ready(): CRITICAL - Could not import game_services: {e}")
            print(f"[{time.ctime()}] Ensure game_services.py exists in game_app and predinator_core is importable.")
        except Exception as e:
            print(f"[{time.ctime()}] GameAppConfig.ready(): CRITICAL - Error initializing game_services: {e}")
            import traceback
            traceback.print_exc()