import os
import torch
import torchaudio
from tortoise.api import TextToSpeech
from tortoise.utils.audio import load_voice

from pydub import AudioSegment
#AudioSegment.converter = r"C:/Program Files/ffmpeg-7.1.1-essentials_build/ffmpeg-7.1.1-essentials_build/bin/ffmpeg.exe"
#AudioSegment.ffprobe = r"C:/Program Files/ffmpeg-7.1.1-essentials_build/ffmpeg-7.1.1-essentials_build/bin/ffprobe.exe"

def create_voice_over(text, gender='female', music_path=None):
    # Check if a GPU is available and set the device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Initialize the Tortoise TTS model and pass the device directly here
    tts = TextToSpeech(use_deepspeed=False, kv_cache=True, half=False, device=device)

    # Select voice
    if gender.lower() == 'male':
        voice = 'daniel'
    else:
        voice = 'angie'

    # Load the conditioning latents for the selected voice
    voice_samples, conditioning_latents = load_voice(voice)

    # Generate speech
    gen = tts.tts_with_preset(
        text,
        voice_samples=voice_samples,
        conditioning_latents=conditioning_latents,
        preset='fast'
    )

    # Save the generated audio to a file
    try:
        output_path = "output_tortoise.wav"
        torchaudio.save(output_path, gen.squeeze(0).cpu(), 24000)
        print("Voice over saved as output_tortoise.wav")

        # OPTIONAL: Add background music
        if music_path and os.path.exists(music_path):
            print("Adding background music...")
            add_background_music(output_path, music_path)
        else:
            print("No background music added.")

    except Exception as e:
        print(f"An error occurred while saving the audio file: {e}")
        print("Please ensure you have the 'soundfile' library installed and its dependencies are met.")

def add_background_music(voice_path, music_path, output_path="output_with_music.wav", music_volume_db=-20):
    # Load audio and background music using pydub
    voice = AudioSegment.from_wav(voice_path)
    music = AudioSegment.from_file(music_path)

    # Adjust background music volume
    music = music - abs(music_volume_db)

    # Loop background music if it's shorter than voice
    if len(music) < len(voice):
        times = int(len(voice) / len(music)) + 1
        music = music * times

    # Trim music to the length of the voice
    music = music[:len(voice)]

    # Overlay the music under the voice
    combined = voice.overlay(music)

    # Export the final audio
    combined.export(output_path, format="wav")
    print(f"Final audio with music saved as {output_path}")

if __name__ == "__main__":
    text = "ohoom, Hello, this is a high quality voice over using Tortoise TTS. This is awesome! awesome! awesome!"
    gender = "female"
    background_music_path = "bgmusic.mp3"  # Replace with your music file path

    create_voice_over(text, gender, background_music_path)
