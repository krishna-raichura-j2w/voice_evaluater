"""Azure OpenAI LLM integration for text-based analysis"""

from openai import AzureOpenAI
from typing import Dict, Any
from .config import Config


class AzureLLMAnalyzer:
    """Azure OpenAI analyzer for grammar, content, and lingui                                                                                                                                                                                                         stic quality assessment"""
    
    def __init__(self):
        """Initialize Azure OpenAI client"""
        Config.validate()
        
        self.client = AzureOpenAI(
            api_key=Config.AZURE_OPENAI_KEY,
            api_version=Config.AZURE_OPENAI_API_VERSION,
            azure_endpoint=Config.AZURE_OPENAI_ENDPOINT
        )
        
        self.deployment = Config.AZURE_OPENAI_DEPLOYMENT
    
    def analyze_transcript(self, transcript: str, context: str = "") -> Dict[str, Any]:
        """
        Analyze transcript for linguistic quality using LLM
        
        Args:
            transcript: Transcribed text from speech
            context: Optional context about the speech (topic, purpose, etc.)
            
        Returns:
            Dictionary containing:
            - grammar_quality: Grammar assessment score (0-100)
            - sentence_formation: Sentence structure quality (0-100)
            - content_depth: Depth and substance of content (0-100)
            - answer_relevance: Relevance to topic/question (0-100)
            - professional_tone: Professional tone assessment (0-100)
            - feedback: Detailed textual feedback
            - overall_score: Overall linguistic quality (0-100)
        """
        
        # If no transcript, use basic heuristics
        if not transcript or len(transcript.strip()) < 10:
            return self._get_default_scores(transcript)
        
        prompt = self._build_analysis_prompt(transcript, context)
        
        try:
            response = self.client.chat.completions.create(
                model=self.deployment,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an expert linguistic analyst and language teacher. "
                            "Analyze the provided speech transcript and provide detailed, "
                            "constructive assessment of its quality across multiple dimensions."
                        )
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_completion_tokens=1500
            )
            
            analysis = response.choices[0].message.content
            
            # Parse the structured response
            result = self._parse_analysis(analysis)
            
            # If parsing failed to get meaningful scores, use heuristics
            if all(result[k] == 0 for k in ["grammar_quality", "sentence_formation", "content_depth"]):
                return self._calculate_heuristic_scores(transcript, analysis)
            
            return result
            
        except Exception as e:
            print(f"Warning: Azure OpenAI request failed: {str(e)}")
            # Fallback to heuristic analysis
            return self._calculate_heuristic_scores(transcript, "LLM analysis unavailable")
    
    def _get_default_scores(self, transcript: str) -> Dict[str, Any]:
        """Return default scores when no transcript is available"""
        return {
            "grammar_quality": 0,
            "sentence_formation": 0,
            "content_depth": 0,
            "answer_relevance": 0,
            "professional_tone": 0,
            "overall_score": 0,
            "feedback": "No transcript available for analysis"
        }
    
    def _calculate_heuristic_scores(self, transcript: str, feedback: str = "") -> Dict[str, Any]:
        """Calculate scores based on simple linguistic heuristics"""
        
        # Basic metrics
        words = transcript.split()
        word_count = len(words)
        sentences = [s.strip() for s in transcript.split('.') if s.strip()]
        sentence_count = len(sentences)
        
        # Grammar quality (based on sentence structure)
        avg_sentence_length = word_count / max(sentence_count, 1)
        grammar_score = min(70 + (avg_sentence_length - 5) * 2, 100)
        grammar_score = max(grammar_score, 40)  # Floor at 40
        
        # Sentence formation (based on variety)
        unique_words = len(set([w.lower() for w in words]))
        lexical_diversity = (unique_words / word_count) if word_count > 0 else 0
        sentence_score = min(60 + lexical_diversity * 40, 100)
        
        # Content depth (based on word count and complexity)
        content_score = min(50 + word_count * 0.5, 100)
        
        # Answer relevance (moderate default)
        relevance_score = 70
        
        # Professional tone (based on word length average)
        avg_word_length = sum(len(w) for w in words) / max(word_count, 1)
        tone_score = min(60 + (avg_word_length - 4) * 5, 100)
        tone_score = max(tone_score, 50)
        
        overall = (grammar_score + sentence_score + content_score + relevance_score + tone_score) / 5
        
        return {
            "grammar_quality": round(grammar_score, 1),
            "sentence_formation": round(sentence_score, 1),
            "content_depth": round(content_score, 1),
            "answer_relevance": round(relevance_score, 1),
            "professional_tone": round(tone_score, 1),
            "overall_score": round(overall, 1),
            "feedback": feedback if feedback else "Analysis based on linguistic heuristics"
        }
    
    def _build_analysis_prompt(self, transcript: str, context: str) -> str:
        """Build the analysis prompt"""
        
        context_section = f"\n\n**Context/Topic:** {context}" if context else ""
        
        return f"""
Analyze the following speech transcript across multiple dimensions:

**Transcript:**
"{transcript}"
{context_section}

Please provide a detailed assessment with scores (0-100) for each category:

1. **Grammar Quality** (0-100): Assess grammatical correctness, proper verb tenses, subject-verb agreement, article usage, etc.

2. **Sentence Formation** (0-100): Evaluate sentence structure, complexity, coherence, and variety.

3. **Content Depth** (0-100): Assess the depth, substance, and insight of the content. Is it superficial or does it demonstrate understanding?

4. **Answer Relevance** (0-100): How relevant and on-topic is the content? Does it address the subject appropriately?

5. **Professional Tone** (0-100): Evaluate the formality, appropriateness, and professional quality of the language used.

Please structure your response EXACTLY as follows:

SCORES:
Grammar Quality: [score]/100
Sentence Formation: [score]/100
Content Depth: [score]/100
Answer Relevance: [score]/100
Professional Tone: [score]/100
Overall Score: [score]/100

FEEDBACK:
[Provide detailed, constructive feedback here. Include:
- Specific strengths
- Areas for improvement
- Concrete examples from the transcript
- Actionable suggestions for enhancement]
"""
    
    def _parse_analysis(self, analysis: str) -> Dict[str, Any]:
        """Parse the LLM response into structured data"""
        
        result = {
            "grammar_quality": 0,
            "sentence_formation": 0,
            "content_depth": 0,
            "answer_relevance": 0,
            "professional_tone": 0,
            "overall_score": 0,
            "feedback": ""
        }
        
        try:
            # Split into sections
            parts = analysis.split("FEEDBACK:", 1)
            
            if len(parts) == 2:
                scores_section = parts[0]
                result["feedback"] = parts[1].strip()
            else:
                scores_section = analysis
                # Use first 500 chars as feedback
                result["feedback"] = analysis[:500] if len(analysis) > 500 else analysis
            
            # Extract scores
            score_mappings = {
                "Grammar Quality": "grammar_quality",
                "Sentence Formation": "sentence_formation",
                "Content Depth": "content_depth",
                "Answer Relevance": "answer_relevance",
                "Professional Tone": "professional_tone",
                "Overall Score": "overall_score"
            }
            
            import re
            for label, key in score_mappings.items():
                # Look for pattern "Label: XX/100" or "Label: XX"
                pattern = rf"{label}:\s*(\d+(?:\.\d+)?)"
                match = re.search(pattern, scores_section, re.IGNORECASE)
                if match:
                    result[key] = float(match.group(1))
            
            # If no scores found at all, try to extract any numbers that look like scores
            if all(result[k] == 0 for k in ["grammar_quality", "sentence_formation", "content_depth"]):
                # Look for any pattern like "XX/100" in the text
                score_matches = re.findall(r'(\d+(?:\.\d+)?)/100', analysis)
                if score_matches:
                    scores = [float(s) for s in score_matches]
                    if len(scores) >= 5:
                        result["grammar_quality"] = scores[0]
                        result["sentence_formation"] = scores[1]
                        result["content_depth"] = scores[2]
                        result["answer_relevance"] = scores[3]
                        result["professional_tone"] = scores[4]
            
            # If overall score not provided, calculate average
            if result["overall_score"] == 0:
                scores = [
                    result["grammar_quality"],
                    result["sentence_formation"],
                    result["content_depth"],
                    result["answer_relevance"],
                    result["professional_tone"]
                ]
                non_zero_scores = [s for s in scores if s > 0]
                result["overall_score"] = sum(non_zero_scores) / len(non_zero_scores) if non_zero_scores else 0
                
        except Exception as e:
            # Fallback: return the raw analysis as feedback
            result["feedback"] = analysis[:500] if len(analysis) > 500 else analysis
            print(f"Warning: Could not parse LLM response structure: {e}")
        
        return result
    
    def get_improvement_suggestions(self, transcript: str, analysis_results: Dict[str, Any]) -> str:
        """
        Get specific improvement suggestions based on analysis
        
        Args:
            transcript: Original transcript
            analysis_results: Results from analyze_transcript
            
        Returns:
            Detailed improvement suggestions
        """
        
        weak_areas = []
        for area in ["grammar_quality", "sentence_formation", "content_depth", 
                     "answer_relevance", "professional_tone"]:
            if analysis_results.get(area, 100) < 70:
                weak_areas.append(area.replace("_", " ").title())
        
        if not weak_areas:
            return "Excellent work! Your speech demonstrates high quality across all dimensions."
        
        prompt = f"""
Given this transcript: "{transcript}"

The speaker needs improvement in: {', '.join(weak_areas)}

Provide 3-5 specific, actionable suggestions for improvement. Be constructive and encouraging.
"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.deployment,
                messages=[
                    {"role": "system", "content": "You are a supportive language coach."},
                    {"role": "user", "content": prompt}
                ],
                max_completion_tokens=500
            )
            
            return response.choices[0].message.content
            
        except Exception:
            return "Continue practicing to improve your speaking skills."
