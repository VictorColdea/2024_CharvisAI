import speech_recognition as sr  # type: ignore
import os
import requests
import requests
from bs4 import BeautifulSoup
import openai
import faiss
import numpy as np
import os
import json
import requests
import webbrowser  # For opening the video URL in a web browser
import os  # For file path handling
import time
import base64

def record_speech():
    '''Records speech. First recognises ambient noise, then records higher noise, 
    then when noise goes back to ambient that is end of person speaking.'''
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
    '''Converts speech audiofile to a text string'''
    rec = sr.Recognizer()
    try:
        text = rec.recognize_google(audio)
        return text
    except sr.UnknownValueError:
        return ""

# Get the current directory where the Python file is stored
current_dir = os.path.dirname(os.path.abspath(__file__))

input_audio = None
while not input_audio:
    # Capture speech input
    input_audio = record_speech()  # Data recorded by microphone

    # Convert captured audio to text if audio was recorded successfully
    if input_audio:
        input_text = speech_to_text(input_audio)
        print("Transcribed Text:", input_text)
    else:
        print("No audio captured, please try again.")


openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    openai.api_key = ""  # Replace with your API key

def scrape_website(url):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/102.0.5005.63 Safari/537.36"
    }

    try:
        page = requests.get(url, headers=headers)
        page.raise_for_status()

        soup = BeautifulSoup(page.text, "lxml")
        text = soup.get_text(separator=" ", strip=True)
        return text
    except requests.exceptions.RequestException as e:
        print(f"Failed to retrieve {url}. Error: {str(e)}")
        return ""

def scrape_multiple_websites(urls):
    content = ""
    for url in urls:
        page_content = scrape_website(url)
        content += page_content + "\n\n"
    return content

def save_scraped_content(content, filename="scraped_content.json"):
    with open(filename, 'w') as f:
        json.dump({"content": content}, f)

def load_scraped_content(filename="scraped_content.json"):
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            data = json.load(f)
            return data.get("content", "")
    return ""

def get_content_from_storage_or_scrape(urls, filename="scraped_content.json"):
    content = load_scraped_content(filename)
    if not content:
        content = scrape_multiple_websites(urls)
        save_scraped_content(content, filename)
    return content

def chunk_content(content, chunk_size=500):
    words = content.split()
    chunks = [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
    return chunks

def get_embeddings(texts):
    response = openai.Embedding.create(
        model="text-embedding-ada-002",
        input=texts
    )
    embeddings = [data['embedding'] for data in response['data']]
    return embeddings

def create_faiss_index(document_embeddings):
    embedding_dim = len(document_embeddings[0])
    index = faiss.IndexFlatL2(embedding_dim)
    index.add(np.array(document_embeddings))
    return index

def retrieve_relevant_documents(query, index, documents, top_k=3):
    query_embedding = get_embeddings([query])[0]
    D, I = index.search(np.array([query_embedding]), top_k)

    if len(I[0]) == 0 or I[0][0] >= len(documents):
        print("No relevant documents found.")
        return []

    return [documents[i] for i in I[0] if i < len(documents)]

def chat_gpt_with_retrieval(prompt, retrieved_docs):
    if not retrieved_docs:
        return "No relevant information found to answer your question."

    context = "\n\n".join(retrieved_docs)

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant. Please keep responses polite, and not exceeding 200 words: answer in as short a manner as is polite."},
            {"role": "user", "content": f"{context}\n\n{prompt}"}
        ]
    )
    return response['choices'][0]['message']['content'].strip()

urls = [
    "https://www.chu.cam.ac.uk/study-here/undergraduate-applications/",
    "https://www.chu.cam.ac.uk/study-here/postgraduate-applications/",
    "https://www.chu.cam.ac.uk/study-here/postgraduate-applications/applications-for-postgraduate-study-at-cambridge/",
    "https://www.chu.cam.ac.uk/study-here/postgraduate-applications/postgraduate-accommodation-at-cambridge-university/",
    "https://www.chu.cam.ac.uk/study-here/postgraduate-applications/mphil-masters-study-at-cambridge/",
    "https://www.chu.cam.ac.uk/study-here/postgraduate-applications/phd-doctoral-research-at-cambridge/",
    "https://www.chu.cam.ac.uk/study-here/postgraduate-applications/postgraduate-funding-opportunities/",
    "https://www.chu.cam.ac.uk/about/sustainability/",
    "https://www.chu.cam.ac.uk/news/people/using-in-house-talent-to-harness-solar-power-at-churchill-college/",
    "https://www.chu.cam.ac.uk/news/news-and-events/the-true-value-of-the-colleges-tree-collection/",
    "https://shop.chu.cam.ac.uk/collections/all",
    "https://shop.chu.cam.ac.uk/collections/all?page=2",
    "https://www.chu.cam.ac.uk/life-at-churchill-college/",
    "https://www.chu.cam.ac.uk/about/working-at-churchill/",
    "https://www.chu.cam.ac.uk/life-at-churchill-college/clubs-and-societies/",
    "https://www.chu.cam.ac.uk/campus-facilities/accommodation-at-churchill-college/",
    "https://www.chu.cam.ac.uk/campus-facilities/",
    "https://www.chu.cam.ac.uk/campus-facilities/grounds-and-gardens-at-churchill-college/",
    "https://www.chu.cam.ac.uk/campus-facilities/sport-at-churchill-college/",
    "https://www.chu.cam.ac.uk/campus-facilities/music-at-churchill-college/",
    "https://www.chu.cam.ac.uk/campus-facilities/college-library/",
    "https://www.chu.cam.ac.uk/campus-facilities/college-dining/menus/",
    "https://www.chu.cam.ac.uk/campus-facilities/college-dining/",
    "https://conferences.chu.cam.ac.uk/our-meeting-rooms/",
    "https://conferences.chu.cam.ac.uk/",
    "https://www.chu.cam.ac.uk/news/news-and-events/meet-our-new-outreach-officer/",
    "https://www.chu.cam.ac.uk/news/grounds-gardens/grounds-gardens-team-joins-the-big-butterfly-count/",
    "https://www.chu.cam.ac.uk/news/news-and-events/churchill-colleges-2024-undergraduate-entrants/"
]

combined_content = get_content_from_storage_or_scrape(urls)

chunks = chunk_content(combined_content)

document_embeddings = get_embeddings(chunks)

index = create_faiss_index(document_embeddings)

user_question = input_text

retrieved_docs = retrieve_relevant_documents(user_question, index, chunks)

answer = chat_gpt_with_retrieval(user_question, retrieved_docs)

print("Assistant:", answer)


# Path to the image
image_path = "" # Replace with the path to the image file

# Step 1: Upload image
url = "https://api.d-id.com/images"
credentials = "... : ..."  # Replace with your actual username and password
encoded_credentials = base64.b64encode(credentials.encode()).decode()
headers = {
    "accept": "application/json",
    "authorization": f"Basic {encoded_credentials}"
}

# Upload the image
files = {"image": ("abc.jpg", open(image_path, "rb"), "image/jpeg")} # Replace "abc.jpg" with the actual filename
response = requests.post(url, files=files, headers=headers)
response_data = response.json()  # Convert the response text to a Python dictionary

# Extract the URL
image_url = response_data.get('url')  # This gets the 'url' from the response

if image_url:
    print(f"Image uploaded successfully: {image_url}")
else:
    print("Failed to upload image")
    exit()

# Step 2: Use the image URL in the talk creation
url = "https://api.d-id.com/talks"
payload = {
    "source_url": image_url,
    "script": {
        "type": "text",
        "subtitles": "false",
        "provider": {
            "type": "microsoft",
            "voice_id": "Sara"
        },
        "input": answer
    },
    "config": {
        "fluent": "false",
        "pad_audio": "0.0"
    }
}

# Create the talk
response = requests.post(url, json=payload, headers=headers)
talk_response = response.json()  # Get the talk response

# Extract the talk ID from the response
talk_id = talk_response.get("id")
if talk_id:
    print(f"Talk ID: {talk_id}")
else:
    print("Failed to retrieve the talk ID.")
    exit()

# Step 3: Polling to check the status of the video
url = f"https://api.d-id.com/talks/{talk_id}"
max_retries = 10
retry_count = 0

while retry_count < max_retries:
    response = requests.get(url, headers=headers)
    talk_data = response.json()

    # Log the entire talk data response for debugging
    print("Talk Data Response:", talk_data)

    # Check the status of the talk
    status = talk_data.get("status")

    if status == "done":  # Check if the talk is done
        # Video is ready
        video_url = talk_data.get('result_url')  # Use 'result_url' for the video URL
        if video_url:
            print(f"Video is ready: {video_url}")

            # You can now download or open the video using this URL
            video_response = requests.get(video_url)

            # Define a local filename for the downloaded video
            video_filename = "downloaded_video.mp4"

            # Save the video content to a file
            with open(video_filename, "wb") as video_file:
                video_file.write(video_response.content)

            print(f"Video downloaded successfully: {video_filename}")

            # Open the video file using the default video player
            video_path = os.path.abspath(video_filename)
            webbrowser.open(video_path)
            break

    elif status in ["created", "started", "pending"]:
        print(f"Video is still being processed... Current status: {status}")
    else:
        print("Unexpected status:", status)
        break

    # Wait before the next check
    time.sleep(5)
    retry_count += 1

if retry_count == max_retries:
    print("Max retries reached. The video may still be processing.")