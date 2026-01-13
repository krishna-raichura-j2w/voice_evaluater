"""Azure Speech SDK integration for pronunciation assessment"""

import azure.cognitiveservices.speech as speechsdk
from typing import Dict, List, Any, Optional
from .config import Config


class AzureSpeechAnalyzer:
    """Azure Speech Services analyzer for pronunciation and fluency assessment"""
    
    def __init__(self):
        """Initialize Azure Speech SDK"""
        Config.validate()
        
        self.speech_config = speechsdk.SpeechConfig(
            subscription=Config.AZURE_SPEECH_KEY,
            region=Config.AZURE_SPEECH_REGION
        )
        
    def analyze_audio(self, audio_file: str, language: str = "en-US") -> Dict[str, Any]:
        """
        Comprehensive audio analysis using Azure Speech SDK
        
        Args:
            audio_file: Path to WAV audio file
            language: Language code (default: en-US)
            
        Returns:
            Dictionary containing:
            - transcript: Recognized text
            - word_timings: List of words with timestamps
            - pronunciation_accuracy: Overall pronunciation score
            - word_errors: List of words with pronunciation issues
            - completeness: Completeness score
            - fluency: Fluency score (micro-fluency)
            - detailed_scores: All available scores
        """
        # Configure audio input
        audio_config = speechsdk.audio.AudioConfig(filename=audio_file)
        
        # Configure pronunciation assessment
        pronunciation_config = speechsdk.PronunciationAssessmentConfig(
            grading_system=speechsdk.PronunciationAssessmentGradingSystem.HundredMark,
            granularity=speechsdk.PronunciationAssessmentGranularity.Phoneme,
            enable_miscue=True
        )
        
        # Enable prosody assessment if available
        try:
            pronunciation_config.enable_prosody_assessment()
        except AttributeError:
            pass
        
        # Create speech recognizer
        speech_recognizer = speechsdk.SpeechRecognizer(
            speech_config=self.speech_config,
            audio_config=audio_config,
            language=language
        )
        
        # Apply pronunciation assessment
        pronunciation_config.apply_to(speech_recognizer)
        
        # Recognize speech
        result = speech_recognizer.recognize_once_async().get()
        
        if result.reason == speechsdk.ResultReason.RecognizedSpeech:
            return self._process_result(result)
        elif result.reason == speechsdk.ResultReason.NoMatch:
            raise ValueError(f"No speech recognized in audio file: {result.no_match_details}")
        elif result.reason == speechsdk.ResultReason.Canceled:
            cancellation = result.cancellation_details
            raise RuntimeError(
                f"Speech recognition canceled: {cancellation.reason}\n"
                f"Error details: {cancellation.error_details}"
            )
        else:
            raise RuntimeError(f"Unexpected recognition result: {result.reason}")
    
    def _process_result(self, result) -> Dict[str, Any]:
        """Process Azure Speech SDK result"""
        pronunciation_result = speechsdk.PronunciationAssessmentResult(result)
        
        # Extract word-level details
        word_timings = []
        word_errors = []
        
        for word in pronunciation_result.words:
            word_info = {
                "word": word.word,
                "accuracy_score": word.accuracy_score,
            }
            
            # Add timing info if available
            if hasattr(word, 'offset'):
                word_info["offset"] = word.offset / 10000000  # Convert to seconds
            if hasattr(word, 'duration'):
                word_info["duration"] = word.duration / 10000000  # Convert to seconds
            
            # Check for pronunciation errors
            if hasattr(word, 'error_type') and word.error_type != 'None':
                word_errors.append({
                    "word": word.word,
                    "error_type": word.error_type,
                    "accuracy_score": word.accuracy_score
                })
            
            word_timings.append(word_info)
        
        # Compile detailed scores
        detailed_scores = {
            "accuracy_score": pronunciation_result.accuracy_score,
            "fluency_score": pronunciation_result.fluency_score,
            "completeness_score": pronunciation_result.completeness_score,
            "pronunciation_score": pronunciation_result.pronunciation_score
        }
        
        # Add prosody score if available
        if hasattr(pronunciation_result, 'prosody_score'):
            detailed_scores["prosody_score"] = pronunciation_result.prosody_score
        
        # Add content assessment if available
        if hasattr(pronunciation_result, 'content_assessment_result'):
            content = pronunciation_result.content_assessment_result
            if hasattr(content, 'grammar_score'):
                detailed_scores["grammar_score"] = content.grammar_score
            if hasattr(content, 'vocabulary_score'):
                detailed_scores["vocabulary_score"] = content.vocabulary_score
            if hasattr(content, 'topic_score'):
                detailed_scores["topic_score"] = content.topic_score
        
        return {
            "transcript": result.text,
            "word_timings": word_timings,
            "pronunciation_accuracy": pronunciation_result.accuracy_score,
            "word_errors": word_errors,
            "completeness": pronunciation_result.completeness_score,
            "fluency": pronunciation_result.fluency_score,
            "prosody_score": detailed_scores.get("prosody_score", 0),
            "detailed_scores": detailed_scores
        }
    
    def simple_transcription(self, audio_file: str, language: str = "en-US") -> str:
        """
        Simple transcription without pronunciation assessment
        
        Args:
            audio_file: Path to audio file
            language: Language code
            
        Returns:
            Transcribed text
        """
        audio_config = speechsdk.audio.AudioConfig(filename=audio_file)
        
        speech_recognizer = speechsdk.SpeechRecognizer(
            speech_config=self.speech_config,
            audio_config=audio_config,
            language=language
        )
        
        result = speech_recognizer.recognize_once_async().get()
        
        if result.reason == speechsdk.ResultReason.RecognizedSpeech:
            return result.text
        else:
            raise RuntimeError(f"Transcription failed: {result.reason}")
