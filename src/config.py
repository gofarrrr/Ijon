# Config
"""
Simple config for testing.
"""

class Settings:
    log_level = "INFO"
    dev_mode = False
    
    def get_log_file_path(self):
        return None

def get_settings():
    return Settings()