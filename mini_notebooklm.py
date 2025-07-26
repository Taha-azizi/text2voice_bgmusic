import os
import torch
import torchaudio
from tortoise.api import TextToSpeech
from tortoise.utils.audio import load_voice

from pydub import AudioSegment
from pydub.silence import detect_silence # To help with combining audio more smoothly
import ollama # For interacting with local LLMs
import re

# --- Configuration ---
# You might need to uncomment and adjust these paths if ffmpeg is not in your system PATH
# AudioSegment.converter = r"C:/Program Files/ffmpeg-7.1.1-essentials_build/ffmpeg-7.1.1-essentials_build/bin/ffmpeg.exe"
# AudioSegment.ffprobe = r"C:/Program Files/ffmpeg-7.1.1-essentials_build/ffmpeg-7.1.1-essentials_build/bin/ffprobe.exe"

def split_text_into_chunks(text, max_length=200):
    """
    Split text into smaller chunks based on sentences and word count.
    Tortoise TTS has a ~400 token limit, so we use 200 words as a safe limit.
    """
    # First, try to split by sentences
    sentences = re.split(r'[.!?]+', text)
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
            
        # Check if adding this sentence would exceed the limit
        test_chunk = current_chunk + " " + sentence if current_chunk else sentence
        word_count = len(test_chunk.split())
        
        if word_count <= max_length:
            current_chunk = test_chunk
        else:
            # If current chunk has content, save it
            if current_chunk:
                chunks.append(current_chunk.strip())
            
            # If single sentence is too long, split by words
            if len(sentence.split()) > max_length:
                words = sentence.split()
                for i in range(0, len(words), max_length):
                    chunk = " ".join(words[i:i + max_length])
                    chunks.append(chunk)
                current_chunk = ""
            else:
                current_chunk = sentence
    
    # Add the last chunk if it has content
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks

# --- Tortoise TTS Setup (modified to handle text chunking) ---
def create_voice_over(text, gender='female', output_filename="temp_audio.wav"):
    """Generates a voice-over using Tortoise TTS with automatic text chunking."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    tts = TextToSpeech(use_deepspeed=False, kv_cache=True, half=False, device=device)

    if gender.lower() == 'male':
        voice = 'daniel'
    elif gender.lower() == 'female':
        voice = 'angie'
    else:
        print(f"Warning: Gender '{gender}' not recognized. Using 'angie'.")
        voice = 'angie'

    voice_samples, conditioning_latents = load_voice(voice)
    
    # Split text into manageable chunks
    text_chunks = split_text_into_chunks(text)
    print(f"Split text into {len(text_chunks)} chunks for TTS processing")
    
    # Generate audio for each chunk
    audio_segments = []
    for i, chunk in enumerate(text_chunks):
        print(f"Generating audio for chunk {i+1}/{len(text_chunks)}")
        try:
            gen = tts.tts_with_preset(
                chunk,
                voice_samples=voice_samples,
                conditioning_latents=conditioning_latents,
                preset='fast'
            )
            audio_segments.append(gen.squeeze(0).cpu())
        except AssertionError as e:
            print(f"Error with chunk {i+1}: {e}")
            print(f"Chunk content: {chunk[:100]}...")
            # Skip this chunk and continue
            continue
        except Exception as e:
            print(f"Unexpected error with chunk {i+1}: {e}")
            continue
    
    if not audio_segments:
        print("No audio segments were generated successfully")
        return None
    
    # Combine all audio segments
    try:
        if len(audio_segments) == 1:
            combined_audio = audio_segments[0]
        else:
            combined_audio = torch.cat(audio_segments, dim=-1)
        
        torchaudio.save(output_filename, combined_audio, 24000)
        print(f"Voice segment saved as {output_filename}")
        return output_filename
    except Exception as e:
        print(f"An error occurred while saving the combined audio file: {e}")
        return None

def add_background_music(voice_path, music_path, output_path="output_with_music.wav", music_volume_db=-20, fade_in_ms=1000, fade_out_ms=1000):
    """Overlays background music onto a voice-over."""
    print("Adding background music...")
    try:
        voice = AudioSegment.from_wav(voice_path)
        music = AudioSegment.from_file(music_path)

        music = music - abs(music_volume_db) # Adjust background music volume

        if len(music) < len(voice): # Loop music if shorter
            times = int(len(voice) / len(music)) + 1
            music = music * times
        music = music[:len(voice)] # Trim music to voice length

        # Apply fades for smoother transition
        music = music.fade_in(fade_in_ms).fade_out(fade_out_ms)
        voice = voice.fade_in(fade_in_ms).fade_out(fade_out_ms)

        combined = voice.overlay(music)
        combined.export(output_path, format="wav")
        print(f"Final audio with music saved as {output_path}")
    except Exception as e:
        print(f"Error adding background music: {e}")
        print("Please ensure ffmpeg is correctly installed and configured for pydub.")

def combine_audio_segments(audio_files, output_path="combined_podcast.wav", silence_duration_ms=500):
    """Combines multiple audio files into a single output file with optional silence."""
    combined_audio = AudioSegment.empty()
    silence = AudioSegment.silent(duration=silence_duration_ms)

    for i, file in enumerate(audio_files):
        if os.path.exists(file):
            segment = AudioSegment.from_wav(file)
            combined_audio += segment
            if i < len(audio_files) - 1: # Add silence between segments
                combined_audio += silence
            os.remove(file) # Clean up temporary files
        else:
            print(f"Warning: Audio file not found for combining: {file}")

    combined_audio.export(output_path, format="wav")
    print(f"All podcast segments combined into {output_path}")
    return output_path

# --- LLM Integration for Podcast Script Generation (modified for shorter segments) ---
def generate_podcast_script(topic, llm_model="gemma3:27b", max_tokens=300):
    """
    Generates a short podcast script on a given topic using a local LLM (Ollama).
    The script will alternate between 'Man:' and 'Woman:' dialogue.
    Modified to generate shorter individual segments.
    """
    print(f"\nGenerating podcast script on '{topic}' using Ollama model '{llm_model}'...")
    prompt = f"""
    You are creating a very short podcast script about '{topic}'.
    The conversation should be between a 'Man' and a 'Woman'.
    Keep each speaker's turn VERY brief - maximum 2-3 sentences per turn.
    Each individual speaking turn should be under 150 words.
    The total conversation should be no more than 300 words.
    Format the output clearly, alternating between 'Man:' and 'Woman:'.

    Example format:
    Man: Hello everyone and welcome to our show! Today we're exploring {topic}.
    Woman: That's right! This is such a fascinating area to dive into.
    Man: Let's start with the basics. What exactly is this about?
    Woman: Great question! It involves several key concepts.
    
    Keep individual turns short and conversational.
    """
    try:
        response = ollama.chat(model=llm_model, messages=[
            {
                'role': 'user',
                'content': prompt,
            },
        ])
        script = response['message']['content']
        print("--- Generated Script ---")
        print(script)
        print("------------------------")
        return script
    except ollama.ResponseError as e:
        print(f"Error connecting to Ollama or generating response: {e}")
        print("Please ensure Ollama is running and the model (e.g., 'gemma3:27b') is downloaded locally.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred during LLM interaction: {e}")
        return None

def parse_script(script_text):
    """Parses the generated script into a list of (gender, text) tuples."""
    segments = []
    lines = script_text.strip().split('\n')
    for line in lines:
        line = line.strip()
        if line.lower().startswith('man:'):
            text = line[4:].strip()
            if text:  # Only add non-empty text
                segments.append(('male', text))
        elif line.lower().startswith('woman:'):
            text = line[6:].strip()
            if text:  # Only add non-empty text
                segments.append(('female', text))
        elif line and segments: # Catch any lines not perfectly formatted
            print(f"Warning: Unformatted line detected, appending to last speaker: {line}")
            segments[-1] = (segments[-1][0], segments[-1][1] + " " + line)
    return segments

# --- Main Podcast Generation Logic ---
def create_podcast_from_topic(topic, llm_model="gemma3:27b", background_music_path="bgmusic.mp3"):
    """
    Generates a full mini-podcast on a given topic using LLM for script and Tortoise TTS for voice.
    """
    script = generate_podcast_script(topic, llm_model)
    if not script:
        print("Failed to generate script. Exiting.")
        return

    parsed_segments = parse_script(script)
    if not parsed_segments:
        print("No valid dialogue segments parsed from script. Exiting.")
        return

    temp_audio_files = []
    print(f"\nGenerating voice-overs for {len(parsed_segments)} segments...")
    for i, (gender, text) in enumerate(parsed_segments):
        if text and text.strip():  # Ensure there's actual text to synthesize
            print(f"\nProcessing segment {i+1}/{len(parsed_segments)} ({gender})")
            print(f"Text: {text[:100]}...")  # Show first 100 chars
            
            output_filename = f"temp_segment_{i}.wav"
            generated_file = create_voice_over(text, gender, output_filename)
            if generated_file:
                temp_audio_files.append(generated_file)
            else:
                print(f"Failed to generate audio for segment {i+1}")

    if not temp_audio_files:
        print("No audio files were generated. Exiting.")
        return

    print(f"\nSuccessfully generated {len(temp_audio_files)} audio segments")

    # Combine all generated audio segments
    final_voice_only_path = "podcast_voice_only.wav"
    combined_audio_path = combine_audio_segments(temp_audio_files, final_voice_only_path)

    # Add background music to the final combined voice track
    if os.path.exists(background_music_path):
        add_background_music(combined_audio_path, background_music_path, output_path=f"podcast_about_{topic.replace(' ', '_').lower()}_with_music.wav")
    else:
        print(f"Background music file not found at {background_music_path}. Final podcast is voice-only.")
        # Rename the voice-only file to a more descriptive name
        final_output = f"podcast_about_{topic.replace(' ', '_').lower()}_voice_only.wav"
        if os.path.exists(final_voice_only_path):
            os.rename(final_voice_only_path, final_output)
            print(f"Voice-only podcast saved as: {final_output}")

    print("\nPodcast generation complete!")

if __name__ == "__main__":
    # --- Set your podcast topic and Ollama model ---
    podcast_topic = "The Impact of AI on Everyday Life"
    ollama_model_name = "gemma3:27b" # Or "gemma:7b" if do not you have enough RAM

    # --- Path to your background music ---
    background_music_path = "bgmusic.mp3" # Make sure this file exists!

    # --- Run the podcast creation ---
    create_podcast_from_topic(podcast_topic, ollama_model_name, background_music_path)