import asyncio
import websockets
import json
import base64
import pyaudio
import os
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Azure configuration
AZURE_REALTIME_ENDPOINT = os.getenv("AZURE_REALTIME_ENDPOINT")
AZURE_API_KEY = os.getenv("AZURE_REALTIME_KEY")

# Audio configuration
SAMPLE_RATE = 24000
CHANNELS = 1
CHUNK_SIZE = 1024
FORMAT = pyaudio.paInt16


class TalkingBot:
    def __init__(self):
        self.ws = None
        self.audio = pyaudio.PyAudio()
        self.is_recording = False
        self.is_running = True

        # Conversation tracking
        self.last_user_transcript = ""
        self.last_ai_response = ""
        self.response_in_progress = False

    async def connect(self):
        """Connect to the Azure OpenAI Realtime API"""
        print("Connecting to Azure OpenAI Realtime API...")

        self.ws = await websockets.connect(
            AZURE_REALTIME_ENDPOINT, additional_headers={"api-key": AZURE_API_KEY}
        )

        print("✓ Connected successfully!\n")

        # Configure session for natural conversation
        instructions = """You are a friendly and helpful AI assistant. Have natural conversations with the user.
        
Key behaviors:
- Be conversational and engaging
- Keep responses concise but informative
- Ask follow-up questions when appropriate
- Be helpful and supportive
- Show personality and warmth in your responses
"""

        session_update = {
            "type": "session.update",
            "session": {
                "modalities": ["text", "audio"],
                "instructions": instructions,
                "voice": "alloy",  # Options: alloy, echo, shimmer
                "input_audio_format": "pcm16",
                "output_audio_format": "pcm16",
                "input_audio_transcription": {"model": "whisper-1"},
                "turn_detection": {
                    "type": "server_vad",
                    "threshold": 0.5,
                    "prefix_padding_ms": 300,
                    "silence_duration_ms": 500,
                },
            },
        }

        await self.ws.send(json.dumps(session_update))
        response = await self.ws.recv()
        print(f"Session configured: {json.loads(response)['type']}\n")

        # Start with a greeting
        await asyncio.sleep(0.5)
        await self.send_message("Say: 'Hello! I'm your AI assistant. How can I help you today?'")

    async def send_message(self, text):
        """Send a text message to trigger AI response"""
        # Wait if there's already a response in progress
        while self.response_in_progress:
            await asyncio.sleep(0.1)

        message = {
            "type": "conversation.item.create",
            "item": {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": text}],
            },
        }
        await self.ws.send(json.dumps(message))

        # Request response
        self.response_in_progress = True
        await self.ws.send(json.dumps({"type": "response.create"}))

    async def send_audio_chunk(self, audio_data):
        """Send audio chunk to the API"""
        audio_b64 = base64.b64encode(audio_data).decode("utf-8")

        event = {"type": "input_audio_buffer.append", "audio": audio_b64}

        await self.ws.send(json.dumps(event))

    async def record_audio(self):
        """Record audio from microphone and send to API"""
        print("🎤 Microphone active... Speak anytime! (Press Ctrl+C to exit)\n")

        stream = self.audio.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=SAMPLE_RATE,
            input=True,
            frames_per_buffer=CHUNK_SIZE,
        )

        self.is_recording = True

        try:
            while self.is_recording and self.is_running:
                audio_data = stream.read(CHUNK_SIZE, exception_on_overflow=False)
                await self.send_audio_chunk(audio_data)
                await asyncio.sleep(0.01)
        except KeyboardInterrupt:
            print("\n✓ Stopping conversation...")
        finally:
            stream.stop_stream()
            stream.close()
            self.is_recording = False

    def play_audio(self, audio_data):
        """Play audio through speakers"""
        try:
            stream = self.audio.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=SAMPLE_RATE,
                output=True,
                frames_per_buffer=CHUNK_SIZE,
            )

            stream.write(audio_data)
            stream.stop_stream()
            stream.close()
        except Exception as e:
            print(f"⚠️  Audio playback error: {e}")

    async def listen_responses(self):
        """Listen for responses from the API"""
        print("=" * 60)
        print("         CONVERSATIONAL AI BOT - READY")
        print("=" * 60)
        print()

        audio_buffer = b""
        text_buffer = []

        try:
            while self.is_running:
                message = await self.ws.recv()
                data = json.loads(message)
                msg_type = data.get("type", "")

                # Handle key events
                if msg_type == "session.updated":
                    print("✓ Session ready - You can start talking!\n")

                elif msg_type == "input_audio_buffer.speech_started":
                    print("🎤 You're speaking...")

                elif msg_type == "input_audio_buffer.speech_stopped":
                    print("✓ Processing your message...\n")

                elif msg_type == "conversation.item.input_audio_transcription.completed":
                    transcript = data.get("transcript", "")
                    self.last_user_transcript = transcript
                    print(f"You: {transcript}")

                elif msg_type == "response.audio_transcript.delta":
                    delta = data.get("delta", "")
                    if delta:
                        text_buffer.append(delta)

                elif msg_type == "response.audio_transcript.done":
                    full_text = "".join(text_buffer)
                    if full_text:
                        self.last_ai_response = full_text
                    text_buffer = []

                elif msg_type == "response.audio.delta":
                    delta_b64 = data.get("delta", "")
                    if delta_b64:
                        audio_buffer += base64.b64decode(delta_b64)

                elif msg_type == "response.audio.done":
                    if audio_buffer:
                        # Play audio
                        self.play_audio(audio_buffer)
                        # Show transcript after playing
                        if self.last_ai_response:
                            print(f"Bot: {self.last_ai_response}")
                        print("─" * 60)
                        print()
                        audio_buffer = b""

                elif msg_type == "response.done":
                    self.response_in_progress = False

                elif msg_type == "error":
                    print(f"❌ Error: {data}")
                    error_message = data.get("error", {})
                    print(f"   Details: {error_message}")

        except websockets.exceptions.WebSocketException as e:
            print(f"WebSocket error: {e}")
        except KeyboardInterrupt:
            print("\n\n✓ Conversation ended by user")

    async def run(self):
        """Main run loop"""
        try:
            await self.connect()

            # Start listening for responses in background
            listen_task = asyncio.create_task(self.listen_responses())

            # Start recording audio
            record_task = asyncio.create_task(self.record_audio())

            # Wait for both tasks
            await asyncio.gather(listen_task, record_task)

        except KeyboardInterrupt:
            print("\n✓ Shutting down...")
        finally:
            self.is_running = False
            self.is_recording = False
            if self.ws:
                await self.ws.close()
            self.audio.terminate()
            print("\n👋 Goodbye!")


async def main():
    """Entry point"""
    print()
    print("=" * 60)
    print("      🤖 AI VOICE CONVERSATION BOT 🤖")
    print("=" * 60)
    print()
    print("Instructions:")
    print("  • Speak naturally when you see '🎤 You're speaking...'")
    print("  • The bot will respond with voice and text")
    print("  • Press Ctrl+C to exit anytime")
    print()

    bot = TalkingBot()
    await bot.run()


if __name__ == "__main__":
    asyncio.run(main())
