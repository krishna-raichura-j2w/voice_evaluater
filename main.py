"""Voice Evaluator - Main orchestration module"""

import sys
import argparse
from pathlib import Path
from typing import Optional

from src.config import Config
from src.utils import convert_to_wav, get_audio_duration, calculate_cost
from src.azure_speech import AzureSpeechAnalyzer
from src.azure_llm import AzureLLMAnalyzer
from src.mti_analyzer import MTIAnalyzer
from src.confidence_analyzer import ConfidenceAnalyzer
from src.report_generator import ReportGenerator


class VoiceEvaluator:
    """Main orchestrator for comprehensive voice evaluation"""
    
    def __init__(self):
        """Initialize all analyzers"""
        print("🚀 Initializing Voice Evaluator...")
        
        # Validate configuration
        Config.validate()
        
        # Initialize components
        self.speech_analyzer = AzureSpeechAnalyzer()
        self.llm_analyzer = AzureLLMAnalyzer()
        self.mti_analyzer = MTIAnalyzer()
        self.confidence_analyzer = ConfidenceAnalyzer()
        self.report_generator = ReportGenerator()
        
        print("✓ All components initialized\n")
    
    def evaluate(self, audio_file: str, context: str = "") -> dict:
        """
        Perform comprehensive voice evaluation
        
        Args:
            audio_file: Path to audio file
            context: Optional context about the speech topic
            
        Returns:
            Complete evaluation report
        """
        
        audio_path = Path(audio_file)
        
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_file}")
        
        print("=" * 80)
        print(f"📁 Analyzing: {audio_path.name}")
        print("=" * 80)
        
        # Convert to WAV if needed
        if audio_path.suffix.lower() != '.wav':
            print(f"\n🔄 Converting {audio_path.suffix} to WAV format...")
            wav_file = convert_to_wav(
                str(audio_path),
                sample_rate=Config.TARGET_SAMPLE_RATE,
                channels=Config.TARGET_CHANNELS
            )
            print(f"✓ Converted to WAV\n")
        else:
            wav_file = str(audio_path)
        
        # Show audio duration and cost estimate
        duration = get_audio_duration(wav_file)
        cost_info = calculate_cost(duration)
        print(f"⏱️  Audio Duration: {cost_info['duration_seconds']:.2f}s ({cost_info['duration_minutes']:.2f} min)")
        print(f"💵 Estimated Cost: ${cost_info['cost_usd']:.6f}\n")
        
        # Step 1: Azure Speech SDK Analysis
        print("🎤 [1/4] Analyzing speech with Azure Speech SDK...")
        try:
            speech_results = self.speech_analyzer.analyze_audio(wav_file)
            print(f"✓ Speech analysis complete")
            print(f"  • Transcript: \"{speech_results['transcript'][:80]}...\"")
            print(f"  • Pronunciation: {speech_results['pronunciation_accuracy']:.1f}/100")
            print(f"  • Fluency: {speech_results['fluency']:.1f}/100")
            if speech_results.get('prosody_score', 0) > 0:
                print(f"  • Prosody: {speech_results['prosody_score']:.1f}/100")
            print()
        except Exception as e:
            print(f"❌ Speech analysis failed: {str(e)}")
            print("   Continuing with default values...\n")
            speech_results = {
                "transcript": "",
                "word_timings": [],
                "pronunciation_accuracy": 0,
                "word_errors": [],
                "completeness": 0,
                "fluency": 0,
                "prosody_score": 0,
                "detailed_scores": {}
            }
        
        # Step 2: Azure LLM Analysis
        print("🤖 [2/4] Analyzing content with Azure OpenAI...")
        try:
            llm_results = self.llm_analyzer.analyze_transcript(
                speech_results['transcript'],
                context=context
            )
            print(f"✓ LLM analysis complete")
            print(f"  • Grammar: {llm_results['grammar_quality']:.1f}/100")
            print(f"  • Content Depth: {llm_results['content_depth']:.1f}/100\n")
        except Exception as e:
            print(f"❌ LLM analysis failed: {str(e)}")
            print("   Continuing with default values...\n")
            llm_results = {
                "grammar_quality": 0,
                "sentence_formation": 0,
                "content_depth": 0,
                "answer_relevance": 0,
                "professional_tone": 0,
                "overall_score": 0,
                "feedback": "Analysis not available"
            }
        
        # Step 3: MTI/Accent Analysis
        print("🌍 [3/4] Analyzing accent and MTI impact...")
        try:
            mti_results = self.mti_analyzer.analyze(wav_file)
            print(f"✓ MTI analysis complete")
            print(f"  • Detected Accent: {mti_results['detected_accent']}")
            print(f"  • MTI Impact: {mti_results['mti_impact_score']:.1f}/100\n")
        except Exception as e:
            print(f"❌ MTI analysis failed: {str(e)}")
            print("   Continuing with default values...\n")
            mti_results = {
                "detected_accent": "unknown",
                "confidence": 0,
                "accent_probabilities": {},
                "mti_impact_score": 50,
                "native_likelihood": 50
            }
        
        # Step 4: Confidence Analysis
        print("💪 [4/4] Analyzing delivery confidence...")
        try:
            confidence_results = self.confidence_analyzer.analyze(
                wav_file,
                speech_results['transcript'],
                speech_results.get('word_timings')
            )
            print(f"✓ Confidence analysis complete")
            print(f"  • Acoustic Confidence: {confidence_results['acoustic_confidence']:.1f}/100")
            print(f"  • Linguistic Confidence: {confidence_results['linguistic_confidence']:.1f}/100\n")
        except Exception as e:
            print(f"❌ Confidence analysis failed: {str(e)}")
            print("   Continuing with default values...\n")
            confidence_results = {
                "acoustic_confidence": 0,
                "linguistic_confidence": 0,
                "overall_confidence": 0,
                "acoustic_features": {},
                "linguistic_features": {}
            }
        
        # Generate final report
        print("📊 Generating comprehensive report...\n")
        report = self.report_generator.generate_report(
            audio_file=audio_file,
            speech_results=speech_results,
            llm_results=llm_results,
            mti_results=mti_results,
            confidence_results=confidence_results
        )
        
        # Print summary
        self.report_generator.print_summary(report)
        
        return report


def main():
    """Main entry point"""
    
    parser = argparse.ArgumentParser(
        description="Voice Evaluator - Comprehensive Audio Assessment System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py audio_files/speech.mp3
  python main.py audio_files/interview.wav --context "Job interview response"
  python main.py audio_files/presentation.m4a --context "Technical presentation"
        """
    )
    
    parser.add_argument(
        "audio_file",
        help="Path to audio file (mp3, wav, m4a, etc.)"
    )
    
    parser.add_argument(
        "--context",
        default="",
        help="Optional context about the speech (topic, purpose, etc.)"
    )
    
    args = parser.parse_args()
    
    try:
        evaluator = VoiceEvaluator()
        evaluator.evaluate(args.audio_file, context=args.context)
        
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
        
    except ValueError as e:
        print(f"❌ Configuration Error: {e}")
        print("\nPlease ensure your .env file is properly configured.")
        print("See .env.example for reference.")
        sys.exit(1)
        
    except Exception as e:
        print(f"❌ Unexpected Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
