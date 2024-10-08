import openai
import json
import requests
import time
import speech_recognition as sr
import os
import pygame

# Set up OpenAI API key
openai.api_key = "" # Replace with your OpenAI API key

def chat_gpt(prompt):
    '''Sends prompt to ChatGPT API to process, and returns the response.'''
    response = openai.ChatCompletion.create(
        model="gpt-4",  # You can use "gpt-3.5-turbo" if preferred
        messages=[
            {"role": "system", "content": "You are a helpful voice assistant at Churchill College in Cambridge. Please keep responses short and to the point, not exceeding 200 words."},
            {"role": "user", "content": prompt}
        ]
    )
    print("Message received")
    return response['choices'][0]['message']['content'].strip()

def record_speech():
    # Initialize recognizer
    recognizer = sr.Recognizer()

    # Open the microphone and start recording
    with sr.Microphone() as source:
        # Calibrate the recognizer to the ambient noise level
        recognizer.adjust_for_ambient_noise(source, duration=1)
        print("Listening for voice input...")

        try:
            # Capture audio until silence is detected
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=5)
            print("Voice detected, processing audio...")
            return audio  # Return after successfully capturing audio
        except sr.WaitTimeoutError:
            print("No voice detected, waiting for input...")
            return None  # Return None if no voice is detected

def speech_to_text(audio):
    '''Converts speech audio to a text string'''
    rec = sr.Recognizer()

    try:
        text = rec.recognize_google(audio)
        return text
    except sr.UnknownValueError:
        return ""

def generate_audio(text):
    '''Generates audio based on input text using ElevenLabs AI'''
    url = "https://api.elevenlabs.io/v1/text-to-speech/..."  # Replace with your ElevenLabs Voice ID
    headers = {
        "accept": "audio/mpeg",
        "xi-api-key": "",  # Replace with your ElevenLabs API key
        "Content-Type": "application/json"
    }
    params = {"optimize_streaming_latency": 0}

    data = {
        "text": text,
        "model_id": "eleven_multilingual_v2",
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.75,
            "style": 0.5,
            "use_speaker_boost": True
        }
    }
    
    response = requests.post(url, headers=headers, params=params, data=json.dumps(data))
    if response.status_code == 200:
        return response.content
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
        quit()

def play_audio(filename):
    '''Plays the audio from a mp3 file (text-to-speech)'''
    pygame.mixer.init()
    pygame.mixer.music.load(filename)
    pygame.mixer.music.play()

    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)
    pygame.mixer.music.unload()

# Find the current directory where the python file is stored
current_dir = os.path.dirname(os.path.abspath(__file__))

# Run the script constantly until closed
while True:
    # Take in speech input from user
    input_speech_data = record_speech()
    if input_speech_data is None:
        continue

    input_text = speech_to_text(input_speech_data)
    print("Input Text: ", input_text)

    # Generate response using OpenAI API
    generated_text = chat_gpt(input_text)
    print("Generated Text: \n\n", generated_text)
    
    # Generate audio file from the response
    response_content = generate_audio(generated_text)

    # Save audio file
    output_speech_filename = f"{current_dir}\\processed_speech.mp3"
    with open(output_speech_filename, 'wb') as f:
        f.write(response_content)
    print(f"Audio saved to {output_speech_filename}")

    # Play the generated audio
    play_audio(output_speech_filename)

    # Delete the audio file after use
    os.remove(output_speech_filename)

    print("Waiting for 5 seconds before next call...")
    time.sleep(5)

