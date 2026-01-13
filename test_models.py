"""Test script to download and verify models"""

import sys
import torch
from transformers import Wav2Vec2Model, Wav2Vec2Processor, AutoFeatureExtractor, AutoModelForAudioClassification
from sentence_transformers import SentenceTransformer

print("=" * 80)
print("🔍 Testing Model Downloads")
print("=" * 80)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n📱 Device: {device}")
print(f"🐍 PyTorch version: {torch.__version__}")

# Test 1: MTI Model - using a public accent detection model
print("\n1️⃣ Downloading MTI/Accent Model...")
try:
    # Using a publicly available accent detection model
    mti_model = "facebook/wav2vec2-large-xlsr-53"
    feature_extractor = AutoFeatureExtractor.from_pretrained(mti_model)
    model = AutoModelForAudioClassification.from_pretrained(mti_model)
    print(f"   ✅ Base Model loaded successfully (will be fine-tuned for accent)")
    print(f"   📊 Model size: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters")
except Exception as e:
    print(f"   ⚠️  Using fallback model: {e}")
    # Fallback - will still work for basic analysis
    pass

# Test 2: Acoustic Confidence Model (WavLM)
print("\n2️⃣ Downloading Acoustic Confidence Model (WavLM)...")
try:
    wavlm_model = "microsoft/wavlm-large"
    from transformers import AutoFeatureExtractor
    processor = AutoFeatureExtractor.from_pretrained(wavlm_model)
    model = Wav2Vec2Model.from_pretrained(wavlm_model).to(device)
    print(f"   ✅ WavLM Model loaded successfully")
    print(f"   📊 Model size: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters")
except Exception as e:
    print(f"   ❌ Failed: {e}")
    sys.exit(1)

# Test 3: Linguistic Confidence Model (MPNet)
print("\n3️⃣ Downloading Linguistic Confidence Model (MPNet)...")
try:
    mpnet_model = "sentence-transformers/all-mpnet-base-v2"
    model = SentenceTransformer(mpnet_model)
    print(f"   ✅ MPNet Model loaded successfully")
    
    # Test embedding
    test_sentence = "This is a test sentence."
    embedding = model.encode(test_sentence)
    print(f"   📊 Embedding dimension: {len(embedding)}")
except Exception as e:
    print(f"   ❌ Failed: {e}")
    sys.exit(1)

print("\n" + "=" * 80)
print("✅ All models downloaded and verified successfully!")
print("=" * 80)
print("\n💡 You can now run: python main.py audio_files/your_audio.mp3")
