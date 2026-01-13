"""Utility functions for audio processing and file handling"""

import json
import subprocess
from pathlib import Path
from typing import Optional


def get_audio_duration(audio_file: str) -> float:
    """Get duration of audio file in seconds using ffprobe"""
    try:
        result = subprocess.run([
            'ffprobe', '-v', 'error',
            '-show_entries', 'format=duration',
            '-of', 'json',
            audio_file
        ], capture_output=True, text=True, check=True)
        
        data = json.loads(result.stdout)
        duration = float(data['format']['duration'])
        return duration
    except Exception:
        return 0.0


def convert_to_wav(input_file: str, output_file: Optional[str] = None, 
                   sample_rate: int = 16000, channels: int = 1) -> str:
    """
    Convert audio file to WAV format optimized for speech recognition
    
    Args:
        input_file: Path to input audio file
        output_file: Path to output WAV file (optional, will auto-generate)
        sample_rate: Target sample rate (default: 16000 Hz)
        channels: Number of channels (default: 1 for mono)
        
    Returns:
        Path to converted WAV file
    """
    if output_file is None:
        input_path = Path(input_file)
        output_file = str(input_path.with_suffix('.wav'))
    
    try:
        subprocess.run([
            'ffmpeg', '-i', input_file,
            '-ar', str(sample_rate),
            '-ac', str(channels),
            '-sample_fmt', 's16',
            '-y',
            output_file
        ], check=True, capture_output=True)
        
        return output_file
        
    except subprocess.CalledProcessError:
        # Fallback to pydub if ffmpeg fails
        try:
            from pydub import AudioSegment
            
            audio = AudioSegment.from_file(input_file)
            audio = audio.set_channels(channels).set_frame_rate(sample_rate).set_sample_width(2)
            audio.export(output_file, format='wav')
            
            return output_file
            
        except ImportError:
            raise RuntimeError(
                "Audio conversion failed. Please install ffmpeg:\n"
                "  sudo apt-get install ffmpeg\n"
                "Or install pydub:\n"
                "  pip install pydub"
            )


def format_score(score: float, total: float = 100) -> str:
    """Format score as percentage with color indicator"""
    percentage = (score / total) * 100
    
    if percentage >= 80:
        indicator = "🟢"
    elif percentage >= 60:
        indicator = "🟡"
    else:
        indicator = "🔴"
    
    return f"{indicator} {score:.1f}/{total}"


def calculate_cost(duration_seconds: float, cost_per_hour: float = 1.0) -> dict:
    """Calculate cost for audio processing"""
    duration_hours = duration_seconds / 3600
    total_cost = duration_hours * cost_per_hour
    
    return {
        "duration_seconds": duration_seconds,
        "duration_minutes": duration_seconds / 60,
        "duration_hours": duration_hours,
        "cost_usd": total_cost,
        "cost_per_hour": cost_per_hour
    }
