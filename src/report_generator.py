"""Report generation and final scoring"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any
from .config import Config
from .utils import format_score


class ReportGenerator:
    """Generate comprehensive evaluation reports"""
    
    def generate_report(self, 
                       audio_file: str,
                       speech_results: Dict[str, Any],
                       llm_results: Dict[str, Any],
                       mti_results: Dict[str, Any],
                       confidence_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate comprehensive evaluation report
        
        Args:
            audio_file: Path to audio file
            speech_results: Results from Azure Speech SDK
            llm_results: Results from Azure LLM
            mti_results: Results from MTI analyzer
            confidence_results: Results from confidence analyzer
            
        Returns:
            Complete evaluation report with final scores
        """
        
        # Calculate final scores
        final_scores = self._calculate_final_scores(
            speech_results, llm_results, mti_results, confidence_results
        )
        
        # Compile full report
        report = {
            "metadata": {
                "audio_file": audio_file,
                "timestamp": datetime.now().isoformat(),
                "version": "1.0.0"
            },
            "transcript": speech_results.get("transcript", ""),
            "scores": final_scores,
            "detailed_results": {
                "azure_speech": speech_results,
                "llm_analysis": llm_results,
                "mti_analysis": mti_results,
                "confidence_analysis": confidence_results
            }
        }
        
        # Save report to file
        report_path = self._save_report(report, audio_file)
        report["report_path"] = report_path
        
        return report
    
    def _calculate_final_scores(self,
                                speech_results: Dict[str, Any],
                                llm_results: Dict[str, Any],
                                mti_results: Dict[str, Any],
                                confidence_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate final weighted scores"""
        
        # Extract individual scores
        pronunciation = speech_results.get("pronunciation_accuracy", 0)
        fluency = speech_results.get("fluency", 0)
        completeness = speech_results.get("completeness", 0)
        
        grammar = llm_results.get("grammar_quality", 0)
        content = llm_results.get("content_depth", 0)
        relevance = llm_results.get("answer_relevance", 0)
        
        mti_impact = mti_results.get("mti_impact_score", 50)
        native_likelihood = mti_results.get("native_likelihood", 50)
        
        acoustic_conf = confidence_results.get("acoustic_confidence", 0)
        linguistic_conf = confidence_results.get("linguistic_confidence", 0)
        overall_conf = confidence_results.get("overall_confidence", 0)
        
        # Category scores (averages)
        speech_quality = (pronunciation + fluency + completeness) / 3
        linguistic_quality = (grammar + content + relevance) / 3
        accent_score = 100 - mti_impact  # Invert MTI impact (lower impact = higher score)
        confidence_score = overall_conf
        
        # Overall final score (weighted)
        final_score = (
            speech_quality * 0.30 +      # 30% - Pronunciation, fluency, completeness
            linguistic_quality * 0.30 +   # 30% - Grammar, content, relevance
            confidence_score * 0.25 +     # 25% - Overall confidence
            accent_score * 0.15           # 15% - Accent/MTI impact
        )
        
        return {
            "final_score": round(final_score, 2),
            "category_scores": {
                "speech_quality": round(speech_quality, 2),
                "linguistic_quality": round(linguistic_quality, 2),
                "confidence": round(confidence_score, 2),
                "accent_clarity": round(accent_score, 2)
            },
            "component_scores": {
                "pronunciation_accuracy": round(pronunciation, 2),
                "fluency": round(fluency, 2),
                "completeness": round(completeness, 2),
                "grammar_quality": round(grammar, 2),
                "content_depth": round(content, 2),
                "answer_relevance": round(relevance, 2),
                "mti_impact": round(mti_impact, 2),
                "native_likelihood": round(native_likelihood, 2),
                "acoustic_confidence": round(acoustic_conf, 2),
                "linguistic_confidence": round(linguistic_conf, 2)
            },
            "grade": self._get_grade(final_score),
            "performance_level": self._get_performance_level(final_score)
        }
    
    def _get_grade(self, score: float) -> str:
        """Convert score to letter grade"""
        if score >= 90:
            return "A+"
        elif score >= 85:
            return "A"
        elif score >= 80:
            return "A-"
        elif score >= 75:
            return "B+"
        elif score >= 70:
            return "B"
        elif score >= 65:
            return "B-"
        elif score >= 60:
            return "C+"
        elif score >= 55:
            return "C"
        elif score >= 50:
            return "C-"
        else:
            return "D"
    
    def _get_performance_level(self, score: float) -> str:
        """Get performance level description"""
        if score >= 85:
            return "Excellent"
        elif score >= 70:
            return "Good"
        elif score >= 55:
            return "Satisfactory"
        else:
            return "Needs Improvement"
    
    def _save_report(self, report: Dict[str, Any], audio_file: str) -> str:
        """Save report to JSON file"""
        
        # Create report filename based on audio file
        audio_path = Path(audio_file)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"{audio_path.stem}_report_{timestamp}.json"
        report_path = Config.REPORTS_DIR / report_filename
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        return str(report_path)
    
    def print_summary(self, report: Dict[str, Any]):
        """Print a formatted summary of the report"""
        
        print("\n" + "=" * 80)
        print("🎤 VOICE EVALUATION REPORT")
        print("=" * 80)
        
        # Metadata
        metadata = report["metadata"]
        print(f"\n📁 Audio File: {metadata['audio_file']}")
        print(f"⏰ Analyzed: {metadata['timestamp']}")
        
        # Transcript
        print(f"\n📝 Transcript:")
        print(f"  \"{report['transcript']}\"")
        
        # Scores
        scores = report["scores"]
        
        print(f"\n" + "=" * 80)
        print(f"🏆 FINAL SCORE: {format_score(scores['final_score'])} - {scores['grade']} ({scores['performance_level']})")
        print("=" * 80)
        
        # Category scores
        print(f"\n📊 Category Scores:")
        cat_scores = scores["category_scores"]
        print(f"  • Speech Quality:       {format_score(cat_scores['speech_quality'])}")
        print(f"  • Linguistic Quality:   {format_score(cat_scores['linguistic_quality'])}")
        print(f"  • Confidence:           {format_score(cat_scores['confidence'])}")
        print(f"  • Accent Clarity:       {format_score(cat_scores['accent_clarity'])}")
        
        # Component scores
        print(f"\n🔍 Detailed Component Scores:")
        comp = scores["component_scores"]
        
        print(f"\n  Speech & Pronunciation:")
        print(f"    - Pronunciation Accuracy:  {format_score(comp['pronunciation_accuracy'])}")
        print(f"    - Fluency:                 {format_score(comp['fluency'])}")
        print(f"    - Completeness:            {format_score(comp['completeness'])}")
        
        print(f"\n  Language & Content:")
        print(f"    - Grammar Quality:         {format_score(comp['grammar_quality'])}")
        print(f"    - Content Depth:           {format_score(comp['content_depth'])}")
        print(f"    - Answer Relevance:        {format_score(comp['answer_relevance'])}")
        
        print(f"\n  Accent & Clarity:")
        print(f"    - MTI Impact:              {format_score(comp['mti_impact'], 100)} (lower is better)")
        print(f"    - Native Likelihood:       {format_score(comp['native_likelihood'])}")
        
        print(f"\n  Confidence:")
        print(f"    - Acoustic Confidence:     {format_score(comp['acoustic_confidence'])}")
        print(f"    - Linguistic Confidence:   {format_score(comp['linguistic_confidence'])}")
        
        # MTI Insights
        mti_results = report["detailed_results"]["mti_analysis"]
        print(f"\n🌍 Accent Analysis:")
        print(f"  • Detected: {mti_results['detected_accent']} (confidence: {mti_results['confidence']*100:.1f}%)")
        
        # LLM Feedback
        llm_results = report["detailed_results"]["llm_analysis"]
        if llm_results.get("feedback"):
            print(f"\n💬 AI Feedback:")
            feedback_lines = llm_results["feedback"].split('\n')
            for line in feedback_lines[:5]:  # Show first 5 lines
                if line.strip():
                    print(f"  {line.strip()}")
        
        # Word errors
        speech_results = report["detailed_results"]["azure_speech"]
        word_errors = speech_results.get("word_errors", [])
        if word_errors:
            print(f"\n⚠️  Words Needing Attention ({len(word_errors)}):")
            for error in word_errors[:5]:  # Show first 5
                print(f"  • '{error['word']}' - {error['error_type']} (score: {error['accuracy_score']:.1f})")
        
        print(f"\n📄 Full report saved to: {report['report_path']}")
        print("=" * 80 + "\n")
