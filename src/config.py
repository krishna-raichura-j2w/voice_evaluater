"""Configuration management for Voice Evaluator"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Config:
    """Central configuration class"""
    
    # Project paths
    BASE_DIR = Path(__file__).parent.parent
    AUDIO_DIR = BASE_DIR / "audio_files"
    REPORTS_DIR = BASE_DIR / "reports"
    
    # Azure Speech Services
    AZURE_SPEECH_KEY = os.getenv("AZURE_SPEECH_KEY")
    AZURE_SPEECH_REGION = os.getenv("AZURE_SPEECH_REGION")
    
    # Azure OpenAI / Chat LLM
    AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
    AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY")
    AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")
    AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
    
    # HuggingFace
    HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
    
    # Model configurations
    MTI_MODEL = "Jzuluaga/accent-id-commonaccent_xlsr-en-english"
    ACOUSTIC_CONFIDENCE_MODEL = "microsoft/wavlm-large"
    LINGUISTIC_CONFIDENCE_MODEL = "sentence-transformers/all-mpnet-base-v2"
    
    # Audio processing settings
    TARGET_SAMPLE_RATE = 16000
    TARGET_CHANNELS = 1
    
    @classmethod
    def validate(cls):
        """Validate that required configuration is present"""
        required = [
            ("AZURE_SPEECH_KEY", cls.AZURE_SPEECH_KEY),
            ("AZURE_SPEECH_REGION", cls.AZURE_SPEECH_REGION),
            ("AZURE_OPENAI_ENDPOINT", cls.AZURE_OPENAI_ENDPOINT),
            ("AZURE_OPENAI_KEY", cls.AZURE_OPENAI_KEY),
            ("AZURE_OPENAI_DEPLOYMENT", cls.AZURE_OPENAI_DEPLOYMENT),
        ]
        
        missing = [name for name, value in required if not value]
        
        if missing:
            raise ValueError(
                f"Missing required configuration: {', '.join(missing)}\n"
                "Please check your .env file."
            )
        
        # Create directories if they don't exist
        cls.AUDIO_DIR.mkdir(exist_ok=True)
        cls.REPORTS_DIR.mkdir(exist_ok=True)
        
        return True
