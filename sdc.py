import asyncio
import websockets
import json
import base64
import pyaudio
import os
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

        self.last_user_transcript = ""
        self.last_ai_response = ""
        self.response_in_progress = False

    async def connect(self):
        """Connect to the Azure OpenAI Realtime API"""
        print("Connecting to Azure OpenAI Realtime API...")

        self.ws = await websockets.connect(
            AZURE_REALTIME_ENDPOINT,
            additional_headers={"api-key": AZURE_API_KEY}
        )

        print("✓ Connected successfully!\n")

        # Interview-driven system prompt
        instructions = """
You are an AI interviewer.

You must follow this flow exactly. Do NOT add anything extra.

Opening rules:
- dont ask any questions how are you and all nothing like that.
- Start with ONLY: "Hello."
- Immediately ask: "Are you free for a short interview now?"
- Do NOT say "How are you?" or any other greeting text.

Flow:

1. Say exactly:
   "Hello. Are you free for a short interview now?"

2. If the user agrees (yes / ok / sure / ready / any positive response):
   - Ask: "What is your name?"
   - After the user gives their name, ask:
     "What is your technical or professional domain?"

3. After the user provides their domain:
   - Ask exactly 3 technical interview questions from that domain.
   - Ask them one by one.
   - Wait for the user’s answer before asking the next question.
   - Do not ask more than 3 questions.

4. After the 3rd answer:
   - Thank the user by name.
   - Say that the interview is complete.
   - End politely.
   - Do not ask any more questions.

Rules:
- Do not say anything outside this flow.
- Do not add small talk.
- Do not say "How are you?"
- Do not restart the interview.
- Do not ask follow-up questions.
- Be concise and professional.
"""

        session_update = {
            "type": "session.update",
            "session": {
                "modalities": ["text", "audio"],
                "instructions": instructions,
                "voice": "alloy",
                "input_audio_format": "pcm16",
                "output_audio_format": "pcm16",
                "input_audio_transcription": {
                    "model": "whisper-1"
                },
                "turn_detection": {
                    "type": "server_vad",
                    "threshold": 0.5,
                    "prefix_padding_ms": 300,
                    "silence_duration_ms": 500
                }
            }
        }

        await self.ws.send(json.dumps(session_update))
        response = await self.ws.recv()
        print(f"Session configured: {json.loads(response)['type']}\n")

        await asyncio.sleep(0.5)
        await self.send_message("Start.")

    async def send_message(self, text):
        while self.response_in_progress:
            await asyncio.sleep(0.1)

        message = {
            "type": "conversation.item.create",
            "item": {
                "type": "message",
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": text
                    }
                ]
            }
        }
        await self.ws.send(json.dumps(message))

        self.response_in_progress = True
        await self.ws.send(json.dumps({"type": "response.create"}))

    async def send_audio_chunk(self, audio_data):
        audio_b64 = base64.b64encode(audio_data).decode("utf-8")
        event = {
            "type": "input_audio_buffer.append",
            "audio": audio_b64
        }
        await self.ws.send(json.dumps(event))

    async def record_audio(self):
        print("🎤 Microphone active... Speak anytime! (Press Ctrl+C to exit)\n")

        stream = self.audio.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=SAMPLE_RATE,
            input=True,
            frames_per_buffer=CHUNK_SIZE
        )

        self.is_recording = True

        try:
            while self.is_recording and self.is_running:
                audio_data = stream.read(CHUNK_SIZE, exception_on_overflow=False)
                await self.send_audio_chunk(audio_data)
                await asyncio.sleep(0.01)
        finally:
            stream.stop_stream()
            stream.close()
            self.is_recording = False

    def play_audio(self, audio_data):
        try:
            stream = self.audio.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=SAMPLE_RATE,
                output=True,
                frames_per_buffer=CHUNK_SIZE
            )

            stream.write(audio_data)
            stream.stop_stream()
            stream.close()
        except Exception as e:
            print(f"⚠️  Audio playback error: {e}")

    async def listen_responses(self):
        print("=" * 60)
        print("         AI INTERVIEW BOT - READY")
        print("=" * 60)
        print()

        audio_buffer = b""
        text_buffer = []

        try:
            while self.is_running:
                message = await self.ws.recv()
                data = json.loads(message)
                msg_type = data.get("type", "")

                if msg_type == "conversation.item.input_audio_transcription.completed":
                    transcript = data.get("transcript", "")
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
                        self.play_audio(audio_buffer)
                        if self.last_ai_response:
                            print(f"Bot: {self.last_ai_response}")
                        print("─" * 60)
                        print()
                        audio_buffer = b""

                elif msg_type == "response.done":
                    self.response_in_progress = False

        except KeyboardInterrupt:
            print("\n✓ Conversation ended by user")

    async def run(self):
        try:
            await self.connect()
            listen_task = asyncio.create_task(self.listen_responses())
            record_task = asyncio.create_task(self.record_audio())
            await asyncio.gather(listen_task, record_task)
        finally:
            self.is_running = False
            self.is_recording = False
            if self.ws:
                await self.ws.close()
            self.audio.terminate()
            print("\n👋 Goodbye!")


async def main():
    print("=" * 60)
    print("      🤖 AI VOICE INTERVIEW BOT 🤖")
    print("=" * 60)
    print()

    bot = TalkingBot()
    await bot.run()


if __name__ == "__main__":
    asyncio.run(main())
