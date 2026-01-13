# Voice Evaluater - Comprehensive Audio Assessment System

A sophisticated voice evaluation system that analyzes speech across multiple dimensions using Azure AI services, HuggingFace models, and advanced acoustic analysis.

## 🎯 Features

### Multi-Model Analysis Pipeline

```
Audio Input
  │
  ├─ 🎤 Azure Speech SDK
  │    ├─ Transcript
  │    ├─ Word timings
  │    ├─ Pronunciation accuracy
  │    ├─ Word errors
  │    ├─ Completeness
  │    └─ Micro-fluency
  │
  ├─ 🤖 Azure OpenAI LLM
  │    ├─ Grammar quality
  │    ├─ Sentence formation
  │    ├─ Content depth
  │    ├─ Answer relevance
  │    ├─ Professional tone
  │    └─ AI Feedback
  │
  ├─ 🌍 MTI Accent Analyzer
  │    ├─ Accent detection
  │    └─ MTI impact score
  │
  ├─ 💪 Confidence Analyzer
  │    ├─ Acoustic confidence (WavLM)
  │    │   ├─ Speech rate stability
  │    │   ├─ Pause regularity
  │    │   ├─ Pitch variance
  │    │   ├─ Energy consistency
  │    │   └─ Completion score
  │    └─ Linguistic confidence (MPNet)
  │         ├─ Semantic coherence
  │         └─ Content complexity
  │
  └─ 📊 Final Scoring & Report
```

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- ffmpeg (for audio conversion)
- Azure Speech Services subscription
- Azure OpenAI subscription

### Installation

1. **Clone or navigate to the project:**
```bash
cd /home/full-stack/J2W/voice_evaluater
```

2. **Install dependencies:**
```bash
uv pip install -e .
# or with pip
pip install -e .
```

3. **Install ffmpeg (if not already installed):**
```bash
# Ubuntu/Debian
sudo apt-get install ffmpeg

# macOS
brew install ffmpeg
```

4. **Configure environment variables:**
```bash
cp .env.example .env
# Edit .env with your Azure credentials
```

Required configuration in `.env`:
```env
AZURE_SPEECH_KEY=your_azure_speech_key
AZURE_SPEECH_REGION=your_region  # e.g., eastus
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_KEY=your_openai_key
AZURE_OPENAI_DEPLOYMENT=your_deployment_name  # e.g., gpt-4
```

## 📖 Usage

### Basic Usage

Place your audio file in the `audio_files` directory and run:

```bash
python main.py audio_files/your_audio.mp3
```

### With Context

Provide context about the speech for better analysis:

```bash
python main.py audio_files/interview.wav --context "Job interview response about teamwork"
```

### Supported Audio Formats

- MP3
- WAV
- M4A
- FLAC
- OGG
- Any format supported by ffmpeg

## 📂 Project Structure

```
voice_evaluater/
├── main.py                 # Main orchestrator
├── src/
│   ├── __init__.py
│   ├── config.py          # Configuration management
│   ├── utils.py           # Utility functions
│   ├── azure_speech.py    # Azure Speech SDK integration
│   ├── azure_llm.py       # Azure OpenAI integration
│   ├── mti_analyzer.py    # MTI/Accent analysis
│   ├── confidence_analyzer.py  # Confidence assessment
│   └── report_generator.py    # Report generation
├── audio_files/           # Input audio files (place your files here)
├── reports/               # Generated JSON reports
├── .env                   # Your configuration (create from .env.example)
├── .env.example          # Configuration template
├── pyproject.toml        # Project dependencies
└── README.md             # This file
```

## 🎯 Models Used

### 1. Azure Speech SDK
- **Purpose**: Transcription, pronunciation, fluency
- **Cost**: ~$1/hour (5 hours free/month)

### 2. Azure OpenAI
- **Purpose**: Grammar, content, linguistic quality
- **Model**: GPT-4 or GPT-3.5-Turbo

### 3. Accent Analyzer
- **Model**: `Jzuluaga/accent-id-commonaccent_xlsr-en-english`
- **Purpose**: MTI impact and accent detection
- **Source**: HuggingFace

### 4. Acoustic Confidence
- **Model**: `microsoft/wavlm-large`
- **Purpose**: Delivery stability analysis
- **Features**: Speech rate, pauses, pitch, energy

### 5. Linguistic Confidence
- **Model**: `sentence-transformers/all-mpnet-base-v2`
- **Purpose**: Content quality and coherence
- **Features**: Semantic coherence, complexity

## 📊 Output

The system generates:

1. **Console Summary**: Real-time progress and key metrics
2. **JSON Report**: Detailed analysis saved to `reports/` directory
3. **Comprehensive Scores**:
   - Final Score (0-100)
   - Category Scores (Speech, Linguistic, Confidence, Accent)
   - Component Scores (10+ individual metrics)
   - Letter Grade (A+ to D)
   - Performance Level

### Example Output

```
🏆 FINAL SCORE: 🟢 85.3/100 - A (Excellent)

📊 Category Scores:
  • Speech Quality:       🟢 88.5/100
  • Linguistic Quality:   🟢 84.2/100
  • Confidence:           🟢 82.7/100
  • Accent Clarity:       🟢 86.0/100

🌍 Accent Analysis:
  • Detected: american (confidence: 87.3%)

💬 AI Feedback:
  Your speech demonstrates strong grammar and clear articulation...
```

## 🛠️ Troubleshooting

### Model Download Issues

First-time use will download models (~2-3GB total). Ensure stable internet connection.

### Audio Format Issues

If audio conversion fails, ensure ffmpeg is properly installed:
```bash
ffmpeg -version
```

### Azure Connection Issues

Verify your credentials:
```bash
# Test Azure Speech
python -c "import azure.cognitiveservices.speech as speechsdk; print('Azure SDK OK')"

# Test Azure OpenAI
python -c "from openai import AzureOpenAI; print('OpenAI SDK OK')"
```

## 💡 Tips

- **Best Audio Quality**: Use WAV files with 16kHz, mono for optimal results
- **Context Matters**: Providing context improves LLM analysis accuracy
- **First Run**: Initial model downloads may take several minutes
- **Cost Tracking**: Check `reports/` for cost estimates per analysis

## 📝 License

This project uses Azure AI services and HuggingFace models. Ensure compliance with their respective licenses.

## 🤝 Contributing

This is a custom evaluation system. For improvements or issues, modify the relevant module in `src/`.

## 📧 Support

For Azure-related issues:
- [Azure Speech Documentation](https://docs.microsoft.com/azure/cognitive-services/speech-service/)
- [Azure OpenAI Documentation](https://docs.microsoft.com/azure/cognitive-services/openai/)

For model-specific questions:
- [HuggingFace Model Hub](https://huggingface.co/models)
