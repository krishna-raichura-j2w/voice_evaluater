"""Confidence Analysis using Acoustic and Linguistic models"""

import torch
import librosa
import numpy as np
from typing import Dict, Any, List, Tuple
from transformers import Wav2Vec2Model, AutoFeatureExtractor
from sentence_transformers import SentenceTransformer
from .config import Config


class ConfidenceAnalyzer:
    """
    Dual-model confidence analyzer:
    1. Acoustic confidence (delivery stability) using WavLM
    2. Linguistic confidence (content quality) using sentence-transformers
    """
    
    def __init__(self):
        """Initialize confidence models"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"Loading confidence models on {self.device}...")
        
        # Acoustic model: microsoft/wavlm-large
        print("  Loading acoustic model (WavLM)...")
        self.acoustic_processor = AutoFeatureExtractor.from_pretrained(
            Config.ACOUSTIC_CONFIDENCE_MODEL
        )
        self.acoustic_model = Wav2Vec2Model.from_pretrained(
            Config.ACOUSTIC_CONFIDENCE_MODEL
        ).to(self.device)
        self.acoustic_model.eval()
        
        # Linguistic model: sentence-transformers/all-mpnet-base-v2
        print("  Loading linguistic model (MPNet)...")
        self.linguistic_model = SentenceTransformer(Config.LINGUISTIC_CONFIDENCE_MODEL)
        
        print("✓ Confidence models loaded successfully")
    
    def analyze(self, audio_file: str, transcript: str, 
                word_timings: List[Dict] = None) -> Dict[str, Any]:
        """
        Comprehensive confidence analysis
        
        Args:
            audio_file: Path to audio file
            transcript: Transcribed text
            word_timings: Optional word timings from Azure Speech SDK
            
        Returns:
            Dictionary containing:
            - acoustic_confidence: Delivery stability score (0-100)
            - linguistic_confidence: Content confidence score (0-100)
            - overall_confidence: Combined confidence score (0-100)
            - acoustic_features: Detailed acoustic analysis
            - linguistic_features: Detailed linguistic analysis
        """
        
        # Analyze acoustic confidence
        acoustic_results = self._analyze_acoustic(audio_file, word_timings)
        
        # Analyze linguistic confidence
        linguistic_results = self._analyze_linguistic(transcript)
        
        # Calculate overall confidence (weighted average)
        overall_confidence = (
            acoustic_results["acoustic_confidence"] * 0.5 +
            linguistic_results["linguistic_confidence"] * 0.5
        )
        
        return {
            "acoustic_confidence": acoustic_results["acoustic_confidence"],
            "linguistic_confidence": linguistic_results["linguistic_confidence"],
            "overall_confidence": overall_confidence,
            "acoustic_features": acoustic_results["features"],
            "linguistic_features": linguistic_results["features"]
        }
    
    def _analyze_acoustic(self, audio_file: str, 
                          word_timings: List[Dict] = None) -> Dict[str, Any]:
        """
        Analyze acoustic confidence (delivery stability)
        
        Extracts:
        - Speech rate stability
        - Pause regularity
        - Pitch variance
        - Energy consistency
        - Sentence-end completion
        """
        
        # Load audio
        audio, sr = librosa.load(audio_file, sr=16000, mono=True)
        
        # Extract prosodic features
        features = {}
        
        # 1. Speech rate stability
        if word_timings:
            speech_rate_stability = self._calculate_speech_rate_stability(word_timings)
            features["speech_rate_stability"] = speech_rate_stability
        else:
            features["speech_rate_stability"] = 0.5  # Neutral default
        
        # 2. Pause regularity
        pause_regularity = self._calculate_pause_regularity(audio, sr)
        features["pause_regularity"] = pause_regularity
        
        # 3. Pitch variance (NOT pitch value, just variance)
        pitch_variance = self._calculate_pitch_variance(audio, sr)
        features["pitch_variance"] = pitch_variance
        
        # 4. Energy consistency
        energy_consistency = self._calculate_energy_consistency(audio)
        features["energy_consistency"] = energy_consistency
        
        # 5. Sentence completion (using WavLM embeddings)
        completion_score = self._calculate_completion_score(audio)
        features["completion_score"] = completion_score
        
        # Calculate acoustic confidence score (0-100)
        acoustic_confidence = (
            features["speech_rate_stability"] * 20 +
            features["pause_regularity"] * 20 +
            features["pitch_variance"] * 20 +
            features["energy_consistency"] * 20 +
            features["completion_score"] * 20
        )
        
        return {
            "acoustic_confidence": float(np.clip(acoustic_confidence, 0, 100)),
            "features": features
        }
    
    def _calculate_speech_rate_stability(self, word_timings: List[Dict]) -> float:
        """Calculate stability of speech rate (0-1)"""
        
        if not word_timings or len(word_timings) < 3:
            return 0.5
        
        # Calculate words per second for each segment
        rates = []
        for i in range(len(word_timings) - 1):
            # Check if duration is available
            if "duration" in word_timings[i]:
                duration = word_timings[i]["duration"]
                if duration > 0:
                    rates.append(1.0 / duration)
        
        if not rates or len(rates) < 2:
            return 0.5
        
        # Lower coefficient of variation = more stable
        mean_rate = np.mean(rates)
        std_rate = np.std(rates)
        
        if mean_rate == 0:
            return 0.5
        
        cv = std_rate / mean_rate
        
        # Convert to 0-1 score (lower CV = higher stability)
        stability = 1.0 / (1.0 + cv)
        
        return float(np.clip(stability, 0, 1))
    
    def _calculate_pause_regularity(self, audio: np.ndarray, sr: int) -> float:
        """Calculate regularity of pauses (0-1)"""
        
        # Detect silence/pauses
        intervals = librosa.effects.split(audio, top_db=30)
        
        if len(intervals) < 2:
            return 0.5
        
        # Calculate pause durations
        pause_durations = []
        for i in range(len(intervals) - 1):
            pause_start = intervals[i][1]
            pause_end = intervals[i + 1][0]
            pause_duration = (pause_end - pause_start) / sr
            if pause_duration > 0.1:  # Consider pauses > 100ms
                pause_durations.append(pause_duration)
        
        if not pause_durations:
            return 0.7  # No pauses = somewhat regular
        
        # Lower variance in pause duration = more regular
        mean_pause = np.mean(pause_durations)
        std_pause = np.std(pause_durations)
        
        if mean_pause == 0:
            return 0.5
        
        cv = std_pause / mean_pause
        regularity = 1.0 / (1.0 + cv)
        
        return float(np.clip(regularity, 0, 1))
    
    def _calculate_pitch_variance(self, audio: np.ndarray, sr: int) -> float:
        """Calculate pitch variance (NOT absolute pitch, just variance) (0-1)"""
        
        # Extract pitch using librosa
        pitches, magnitudes = librosa.piptrack(y=audio, sr=sr)
        
        # Get pitch values where magnitude is high
        pitch_values = []
        for t in range(pitches.shape[1]):
            index = magnitudes[:, t].argmax()
            pitch = pitches[index, t]
            if pitch > 0:  # Valid pitch
                pitch_values.append(pitch)
        
        if not pitch_values:
            return 0.5
        
        # Calculate coefficient of variation
        mean_pitch = np.mean(pitch_values)
        std_pitch = np.std(pitch_values)
        
        if mean_pitch == 0:
            return 0.5
        
        cv = std_pitch / mean_pitch
        
        # Moderate variance (CV around 0.1-0.3) is good for confident speech
        # Too low = monotone, too high = erratic
        if 0.1 <= cv <= 0.3:
            score = 1.0
        elif cv < 0.1:
            score = cv / 0.1  # Penalize monotone
        else:
            score = 0.3 / cv  # Penalize excessive variance
        
        return float(np.clip(score, 0, 1))
    
    def _calculate_energy_consistency(self, audio: np.ndarray) -> float:
        """Calculate energy consistency across speech (0-1)"""
        
        # Calculate RMS energy in windows
        frame_length = 2048
        hop_length = 512
        
        rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
        
        if len(rms) == 0:
            return 0.5
        
        # Calculate coefficient of variation
        mean_rms = np.mean(rms)
        std_rms = np.std(rms)
        
        if mean_rms == 0:
            return 0.5
        
        cv = std_rms / mean_rms
        
        # Lower CV = more consistent energy
        consistency = 1.0 / (1.0 + cv)
        
        return float(np.clip(consistency, 0, 1))
    
    def _calculate_completion_score(self, audio: np.ndarray) -> float:
        """Calculate sentence completion quality using WavLM (0-1)"""
        
        # Process audio with WavLM
        inputs = self.acoustic_processor(
            audio,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True
        )
        
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.acoustic_model(**inputs)
            hidden_states = outputs.last_hidden_state
        
        # Analyze final portion of speech (last 20%)
        seq_len = hidden_states.shape[1]
        final_portion = hidden_states[:, int(seq_len * 0.8):, :]
        
        # Calculate variance in final portion
        # Lower variance at the end = more complete/confident ending
        variance = torch.var(final_portion).item()
        
        # Normalize to 0-1 score
        completion = 1.0 / (1.0 + variance * 10)
        
        return float(np.clip(completion, 0, 1))
    
    def _analyze_linguistic(self, transcript: str) -> Dict[str, Any]:
        """
        Analyze linguistic confidence (content quality)
        
        Uses sentence embeddings to assess:
        - Semantic coherence
        - Content complexity
        - Topic consistency
        """
        
        if not transcript or len(transcript.strip()) < 10:
            return {
                "linguistic_confidence": 0,
                "features": {
                    "semantic_coherence": 0,
                    "content_complexity": 0,
                    "sentence_count": 0
                }
            }
        
        # Split into sentences
        sentences = [s.strip() for s in transcript.split('.') if s.strip()]
        
        if not sentences:
            return {
                "linguistic_confidence": 0,
                "features": {
                    "semantic_coherence": 0,
                    "content_complexity": 0,
                    "sentence_count": 0
                }
            }
        
        # Get sentence embeddings
        embeddings = self.linguistic_model.encode(sentences)
        
        features = {}
        
        # 1. Semantic coherence (similarity between consecutive sentences)
        if len(embeddings) > 1:
            coherence_scores = []
            for i in range(len(embeddings) - 1):
                similarity = np.dot(embeddings[i], embeddings[i + 1]) / (
                    np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[i + 1])
                )
                coherence_scores.append(similarity)
            
            semantic_coherence = np.mean(coherence_scores)
            features["semantic_coherence"] = float(np.clip(semantic_coherence, 0, 1))
        else:
            features["semantic_coherence"] = 0.5
        
        # 2. Content complexity (embedding variance)
        if len(embeddings) > 1:
            embedding_variance = np.var(embeddings)
            # Moderate variance indicates complex, diverse content
            content_complexity = 1.0 / (1.0 + np.abs(embedding_variance - 0.5))
            features["content_complexity"] = float(np.clip(content_complexity, 0, 1))
        else:
            features["content_complexity"] = 0.5
        
        # 3. Sentence count (more sentences = more content)
        features["sentence_count"] = len(sentences)
        sentence_score = min(len(sentences) / 5.0, 1.0)  # Normalize to 1 at 5+ sentences
        
        # Calculate linguistic confidence (0-100)
        linguistic_confidence = (
            features["semantic_coherence"] * 40 +
            features["content_complexity"] * 30 +
            sentence_score * 30
        ) * 100
        
        return {
            "linguistic_confidence": float(np.clip(linguistic_confidence, 0, 100)),
            "features": features
        }
