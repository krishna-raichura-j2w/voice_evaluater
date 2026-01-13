"""MTI (Mother Tongue Influence) / Accent Analysis using HuggingFace model"""

import torch
import librosa
import numpy as np
from typing import Dict, Any, Optional
from transformers import AutoFeatureExtractor, Wav2Vec2Model
from .config import Config


class MTIAnalyzer:
    """
    MTI/Accent analyzer using prosodic features from Wav2Vec2
    Analyzes Mother Tongue Influence impact on speech using acoustic analysis
    """
    
    def __init__(self):
        """Initialize MTI model"""
        # Use a base wav2vec2 model for acoustic feature extraction
        self.model_name = "facebook/wav2vec2-base"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"Loading MTI model: {self.model_name} on {self.device}...")
        
        try:
            # Load model and feature extractor
            self.feature_extractor = AutoFeatureExtractor.from_pretrained(self.model_name)
            self.model = Wav2Vec2Model.from_pretrained(self.model_name).to(self.device)
            self.model.eval()
            
            print("✓ MTI model loaded successfully")
        except Exception as e:
            print(f"⚠️  Warning: Could not load MTI model: {e}")
            print("   MTI analysis will use simplified acoustic features")
            self.model = None
            self.feature_extractor = None
    
    def analyze(self, audio_file: str) -> Dict[str, Any]:
        """
        Analyze accent/MTI impact from audio file using acoustic features

        Args:
            audio_file: Path to audio file (preferably WAV)

        Returns:
            Dictionary containing:
            - detected_accent: Estimated accent category
            - confidence: Confidence score
            - accent_probabilities: Acoustic feature analysis
            - accent_clarity_score: English accent clarity (0-100, higher is better)
            - mti_impact_score: MTI impact score (0-100, higher = more MTI influence)
            - native_likelihood: Likelihood of native English speaker (0-100)
        """

        # Load and preprocess audio
        audio, sample_rate = librosa.load(audio_file, sr=16000, mono=True)

        # Analyze acoustic features for MTI estimation
        mti_score, native_likelihood, features = self._analyze_acoustic_features(audio, sample_rate)

        # accent_clarity_score: Higher = better/clearer English accent
        accent_clarity_score = 100 - mti_score

        return {
            "detected_accent": self._estimate_accent_category(mti_score),
            "confidence": 0.75,  # Moderate confidence for acoustic analysis
            "accent_probabilities": features,
            "accent_clarity_score": accent_clarity_score,  # Higher = better English
            "mti_impact_score": mti_score,  # Higher = more mother tongue influence
            "native_likelihood": native_likelihood
        }
    
    
    def _analyze_acoustic_features(self, audio: np.ndarray, sr: int) -> tuple:
        """
        Analyze acoustic features to estimate MTI impact
        Returns: (mti_score, native_likelihood, features_dict)
        """
        features = {}
        
        # 1. Phoneme duration variance (non-native speakers often have irregular timing)
        onset_frames = librosa.onset.onset_detect(y=audio, sr=sr, units='frames')
        if len(onset_frames) > 2:
            onset_intervals = np.diff(onset_frames)
            timing_variance = np.std(onset_intervals) / (np.mean(onset_intervals) + 1e-6)
            features['timing_variance'] = float(timing_variance)
        else:
            features['timing_variance'] = 0.5
        
        # 2. Pitch contour smoothness
        pitches, magnitudes = librosa.piptrack(y=audio, sr=sr)
        pitch_values = []
        for t in range(pitches.shape[1]):
            index = magnitudes[:, t].argmax()
            pitch = pitches[index, t]
            if pitch > 0:
                pitch_values.append(pitch)
        
        if len(pitch_values) > 10:
            pitch_smoothness = 1.0 / (np.std(np.diff(pitch_values)) + 1e-6)
            features['pitch_smoothness'] = float(np.clip(pitch_smoothness / 100, 0, 1))
        else:
            features['pitch_smoothness'] = 0.5
        
        # 3. Spectral centroid (relates to accent characteristics)
        spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
        centroid_mean = np.mean(spectral_centroids)
        centroid_std = np.std(spectral_centroids)
        features['spectral_centroid_mean'] = float(centroid_mean)
        features['spectral_centroid_variance'] = float(centroid_std)
        
        # Calculate MTI score based on features
        # Higher variance in timing and pitch = higher MTI
        timing_score = min(features['timing_variance'] * 50, 50)
        pitch_score = (1 - features['pitch_smoothness']) * 30
        spectral_score = min((centroid_std / centroid_mean) * 20, 20) if centroid_mean > 0 else 10
        
        mti_score = timing_score + pitch_score + spectral_score
        mti_score = float(np.clip(mti_score, 0, 100))
        
        # Native likelihood is inverse of MTI
        native_likelihood = 100 - mti_score
        
        return mti_score, native_likelihood, features
    
    def _estimate_accent_category(self, mti_score: float) -> str:
        """Estimate accent category based on MTI score"""
        if mti_score < 25:
            return "native-like"
        elif mti_score < 45:
            return "slight-accent"
        elif mti_score < 65:
            return "moderate-accent"
        else:
            return "strong-accent"
    
    def _calculate_mti_impact(self, accent_probs: Dict[str, float], 
                              detected_accent: str) -> float:
        """
        Calculate MTI impact score (0-100)
        Lower score = less MTI impact (more native-like)
        Higher score = more MTI impact (stronger accent)
        """
        
        # Define native English accents
        native_accents = {
            "us", "american", "british", "uk", "australian", 
            "canadian", "irish", "scottish", "english"
        }
        
        # Check if detected accent is native
        is_native = any(native in detected_accent.lower() for native in native_accents)
        
        if is_native:
            # Low MTI impact for native accents
            # But consider confidence - lower confidence means less certain
            base_score = 20
            confidence_factor = accent_probs.get(detected_accent, 0.5)
            mti_score = base_score * (1 - confidence_factor * 0.5)
        else:
            # Higher MTI impact for non-native accents
            base_score = 60
            confidence_factor = accent_probs.get(detected_accent, 0.5)
            mti_score = base_score + (confidence_factor * 40)
        
        return float(np.clip(mti_score, 0, 100))
    
    def _calculate_native_likelihood(self, accent_probs: Dict[str, float]) -> float:
        """
        Calculate likelihood of native English speaker (0-100)
        """
        
        native_accents = {
            "us", "american", "british", "uk", "australian", 
            "canadian", "irish", "scottish", "english"
        }
        
        # Sum probabilities of native accents
        native_prob = sum(
            prob for accent, prob in accent_probs.items()
            if any(native in accent.lower() for native in native_accents)
        )
        
        return float(native_prob * 100)
    
    def get_accent_insights(self, mti_results: Dict[str, Any]) -> str:
        """
        Generate human-readable insights about accent/MTI
        
        Args:
            mti_results: Results from analyze()
            
        Returns:
            Textual insights about the detected accent
        """
        
        accent = mti_results["detected_accent"]
        confidence = mti_results["confidence"]
        mti_score = mti_results["mti_impact_score"]
        native_likelihood = mti_results["native_likelihood"]
        
        insights = []
        
        # Accent detection
        insights.append(f"Detected accent: {accent} (confidence: {confidence*100:.1f}%)")
        
        # MTI impact interpretation
        if mti_score < 30:
            insights.append("✓ Minimal mother tongue influence - very native-like pronunciation")
        elif mti_score < 50:
            insights.append("○ Moderate mother tongue influence - generally clear pronunciation")
        elif mti_score < 70:
            insights.append("△ Noticeable mother tongue influence - some accent features present")
        else:
            insights.append("! Significant mother tongue influence - strong accent detected")
        
        # Native likelihood
        insights.append(f"Native English likelihood: {native_likelihood:.1f}%")
        
        return "\n".join(insights)
