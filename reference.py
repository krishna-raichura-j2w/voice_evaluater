#!/usr/bin/env python3
"""
Test Azure Speech SDK with tashini.mp3 audio file
"""

import json
import os
import subprocess
import azure.cognitiveservices.speech as speechsdk


def get_audio_duration(audio_file: str) -> float:
    """Get duration of audio file in seconds"""
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


def show_cost_calculation(audio_file: str):
    """Display cost calculation for the audio processing"""
    duration = get_audio_duration(audio_file)
    
    if duration == 0:
        return
    
    duration_hours = duration / 3600
    cost_per_hour = 1.00  # USD
    total_cost = duration_hours * cost_per_hour
    free_tier_hours = 5.0
    
    print("\n" + "=" * 60)
    print("💰 COST CALCULATION")
    print("=" * 60)
    print(f"\n⏱️  Audio Duration: {duration:.2f} seconds ({duration/60:.2f} minutes)")
    print(f"💵 Cost: ${total_cost:.6f} (${cost_per_hour}/hour)")
    print(f"🎁 Free Tier: {free_tier_hours} hours/month - ✅ THIS IS FREE!")
    print(f"📊 You can process ~{int((free_tier_hours * 3600) / duration)} similar files FREE/month")
    print("=" * 60)


def load_config():
    """Load Azure credentials from config.json"""
    with open("config.json", "r") as f:
        return json.load(f)


def analyze_audio_file(audio_file: str):
    """
    Analyze audio file using Azure Speech Service
    - Transcribe the speech
    - Assess pronunciation quality
    - Evaluate fluency and prosody
    
    Args:
        audio_file: Path to the audio file (tashini.mp3)
    """
    print("=" * 60)
    print(f"Analyzing: {audio_file}")
    print("=" * 60)
    
    config = load_config()
    
    # Configure speech service
    speech_config = speechsdk.SpeechConfig(
        subscription=config["SubscriptionKey"],
        region=config["ServiceRegion"]
    )
    
    # Configure audio input from file
    audio_config = speechsdk.audio.AudioConfig(filename=audio_file)
    
    # Configure pronunciation assessment
    pronunciation_config = speechsdk.PronunciationAssessmentConfig(
        grading_system=speechsdk.PronunciationAssessmentGradingSystem.HundredMark,
        granularity=speechsdk.PronunciationAssessmentGranularity.Phoneme,
        enable_miscue=True
    )
    
    # Enable prosody (rhythm, stress, intonation) if available
    try:
        pronunciation_config.enable_prosody_assessment()
    except AttributeError:
        pass  # Prosody assessment not available in this SDK version
    
    # Create speech recognizer
    speech_recognizer = speechsdk.SpeechRecognizer(
        speech_config=speech_config,
        audio_config=audio_config,
        language="en-US"
    )
    
    # Apply pronunciation assessment
    pronunciation_config.apply_to(speech_recognizer)
    
    print("\n🎤 Processing audio...\n")
    
    # Recognize speech
    result = speech_recognizer.recognize_once_async().get()
    
    # Process results
    if result.reason == speechsdk.ResultReason.RecognizedSpeech:
        print("✓ Speech Recognition Successful!\n")
        print(f"📝 Transcribed Text: '{result.text}'")
        
        # Get pronunciation assessment results
        pronunciation_result = speechsdk.PronunciationAssessmentResult(result)
        
        print("\n" + "=" * 60)
        print("PRONUNCIATION ASSESSMENT SCORES")
        print("=" * 60)
        
        print(f"\n📊 Overall Scores:")
        print(f"  • Accuracy Score:      {pronunciation_result.accuracy_score:.1f}/100")
        print(f"  • Fluency Score:       {pronunciation_result.fluency_score:.1f}/100")
        print(f"  • Completeness Score:  {pronunciation_result.completeness_score:.1f}/100")
        print(f"  • Pronunciation Score: {pronunciation_result.pronunciation_score:.1f}/100")
        
        # Prosody score (if available)
        if hasattr(pronunciation_result, 'prosody_score'):
            print(f"  • Prosody Score:       {pronunciation_result.prosody_score:.1f}/100")
        
        # Content assessment (if available)
        if hasattr(pronunciation_result, 'content_assessment_result'):
            content = pronunciation_result.content_assessment_result
            print(f"\n📚 Content Assessment:")
            if hasattr(content, 'grammar_score'):
                print(f"  • Grammar Score:   {content.grammar_score:.1f}/100")
            if hasattr(content, 'vocabulary_score'):
                print(f"  • Vocabulary Score: {content.vocabulary_score:.1f}/100")
            if hasattr(content, 'topic_score'):
                print(f"  • Topic Score:     {content.topic_score:.1f}/100")
        
        # Word-level details
        print(f"\n📖 Word-by-Word Analysis:")
        print("-" * 60)
        
        words = pronunciation_result.words
        for i, word in enumerate(words, 1):
            print(f"{i}. '{word.word}' - Score: {word.accuracy_score:.1f}/100", end="")
            if hasattr(word, 'error_type'):
                if word.error_type != 'None':
                    print(f" ⚠️  Error: {word.error_type}", end="")
            print()
        
        print("\n" + "=" * 60)
        
    elif result.reason == speechsdk.ResultReason.NoMatch:
        print("✗ No speech could be recognized in the audio file")
        print(f"Details: {result.no_match_details}")
        
    elif result.reason == speechsdk.ResultReason.Canceled:
        cancellation = result.cancellation_details
        print(f"✗ Recognition canceled: {cancellation.reason}")
        if cancellation.reason == speechsdk.CancellationReason.Error:
            print(f"Error details: {cancellation.error_details}")
            print("\nPossible issues:")
            print("  • Check if the audio file exists and is readable")
            print("  • Verify Azure credentials in config.json")
            print("  • Ensure internet connection is active")
            print("  • Check if the audio format is supported")


def simple_transcription(audio_file: str):
    """
    Simple transcription without pronunciation assessment
    
    Args:
        audio_file: Path to the audio file
    """
    print("\n" + "=" * 60)
    print("SIMPLE TRANSCRIPTION (No Assessment)")
    print("=" * 60)
    
    config = load_config()
    
    speech_config = speechsdk.SpeechConfig(
        subscription=config["SubscriptionKey"],
        region=config["ServiceRegion"]
    )
    
    audio_config = speechsdk.audio.AudioConfig(filename=audio_file)
    
    speech_recognizer = speechsdk.SpeechRecognizer(
        speech_config=speech_config,
        audio_config=audio_config,
        language="en-US"
    )
    
    print("\n🎤 Transcribing audio...\n")
    
    result = speech_recognizer.recognize_once_async().get()
    
    if result.reason == speechsdk.ResultReason.RecognizedSpeech:
        print(f"✓ Transcription: '{result.text}'")
    else:
        print(f"✗ Transcription failed: {result.reason}")


def convert_mp3_to_wav(mp3_file: str) -> str:
    """
    Convert MP3 to WAV format using ffmpeg
    
    Args:
        mp3_file: Path to MP3 file
        
    Returns:
        Path to converted WAV file
    """
    wav_file = mp3_file.replace('.mp3', '.wav')
    
    print(f"🔄 Converting {mp3_file} to WAV format...")
    
    try:
        # Convert using ffmpeg - 16kHz mono 16-bit (optimal for speech recognition)
        subprocess.run([
            'ffmpeg', '-i', mp3_file,
            '-ar', '16000',  # Sample rate 16kHz
            '-ac', '1',       # Mono
            '-sample_fmt', 's16',  # 16-bit
            '-y',            # Overwrite output file
            wav_file
        ], check=True, capture_output=True)
        
        print(f"✓ Converted to {wav_file}\n")
        return wav_file
        
    except subprocess.CalledProcessError as e:
        print("✗ ffmpeg conversion failed")
        print("Trying alternative method with pydub...")
        
        try:
            from pydub import AudioSegment
            
            # Load MP3 and convert
            audio = AudioSegment.from_mp3(mp3_file)
            # Convert to mono, 16kHz, 16-bit
            audio = audio.set_channels(1).set_frame_rate(16000).set_sample_width(2)
            audio.export(wav_file, format='wav')
            
            print(f"✓ Converted to {wav_file}\n")
            return wav_file
            
        except ImportError:
            print("\n⚠️  Neither ffmpeg nor pydub is available.")
            print("Please install one of the following:")
            print("  1. ffmpeg: sudo apt-get install ffmpeg")
            print("  2. pydub: pip install pydub")
            raise


def main():
    audio_file = "tashini.mp3"
    
    try:
        # Check if MP3 file exists
        if not os.path.exists(audio_file):
            print(f"✗ Error: {audio_file} not found!")
            return
        
        # Convert MP3 to WAV
        wav_file = convert_mp3_to_wav(audio_file)
        
        # Full analysis with pronunciation assessment
        analyze_audio_file(wav_file)
        
        # Show cost calculation
        show_cost_calculation(wav_file)
        
        print("\n✅ Analysis complete!\n")
        
    except FileNotFoundError as e:
        print(f"\n✗ Error: File not found - {e}")
        print(f"Make sure '{audio_file}' exists in the current directory")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
