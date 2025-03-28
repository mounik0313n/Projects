import streamlit as st
import assemblyai as aai
import time
import sys
import requests
import json
import os

# Set your AssemblyAI API key
aai.settings.api_key = ""  # Replace with your actual AssemblyAI API key

# Initialize the transcriber
transcriber = aai.Transcriber()

# Function for transcription
def transcribe_audio(audio_file_path):
    # Start transcription (this is an asynchronous operation)
    transcript = transcriber.transcribe(audio_file_path)

    # Check the status of the transcription while it's processing
    while transcript.status != 'completed':
        time.sleep(1)  # Wait for 1 second before checking the status again
        transcript = transcriber.get_status(transcript.id)  # Get the latest transcription status

    return transcript.text  # Return the transcription text once complete

# Function for summarization
def summarize_text(api_key, text):
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
    
    headers = {
        "Content-Type": "application/json"
    }
    
    # Prepare the data payload to send to Gemini API
    data = {
        "contents": [{
            "parts": [{"text": text}]
        }]
    }
    
    # Include the API key in the request URL
    response = requests.post(url, headers=headers, params={"key": api_key}, json=data)
    
    if response.status_code == 200:
        return response.json()
    else:
        return f"Error: {response.status_code}, {response.text}"

# Function to clean summary text
def clean_summary(summary):
    # Remove bullet points, markdown syntax, and other special formatting
    cleaned_summary = summary.replace("*", "").replace("**", "").replace("_", "").replace("\n", "\n\n")
    return cleaned_summary.strip()

# Function to save summary to a file
def save_summary_to_file(summary, file_path):
    with open(file_path, 'w') as file:
        file.write(summary)

# Main Streamlit UI
def main():
    st.title("File Transcription and Summarization")

    # Upload any file (audio, text, etc.)
    uploaded_file = st.file_uploader("Upload a file", type=["mp3", "wav", "m4a", "txt", "pdf"])

    if uploaded_file is not None:
        file_extension = uploaded_file.name.split('.')[-1].lower()

        # If the uploaded file is audio
        if file_extension in ['mp3', 'wav', 'm4a']:
            st.audio(uploaded_file, format="audio/wav")  # Display the uploaded audio

            # Button to start the transcription and summarization process
            if st.button('Start Transcription and Summarization'):
                with st.spinner('Transcribing the audio...'):
                    # Save the uploaded file to a temporary location
                    audio_file_path = f"temp_{uploaded_file.name}"
                    with open(audio_file_path, 'wb') as f:
                        f.write(uploaded_file.getbuffer())

                    # Transcribe the audio
                    transcription_text = transcribe_audio(audio_file_path)
                    st.success("Transcription complete!")

                    # Display the transcription result
                    st.subheader("Transcription")
                    st.text_area("Transcribed Text", transcription_text, height=200)

                    # Summarize the transcription using Gemini API
                    st.spinner('Summarizing the transcription...')
                    gemini_api_key = ""  # Replace with your Gemini API key
                    summary_response = summarize_text(gemini_api_key, transcription_text)

                    # Extract summary text from the response
                    if isinstance(summary_response, dict):
                        summary_text = summary_response['candidates'][0]['content']['parts'][0]['text']
                    else:
                        summary_text = summary_response  # If error occurred

                    # Clean the summary text
                    cleaned_summary = clean_summary(summary_text)

                    # Display the summary
                    st.subheader("Summary")
                    st.text_area("Summarized Text", cleaned_summary, height=200)

                    # Option to save the summary to a file
                    summary_file_path = f"summary_{uploaded_file.name}.txt"
                    with open(summary_file_path, 'w') as file:
                        file.write(cleaned_summary)
                    st.download_button(label="Download Summary", data=open(summary_file_path, 'rb'), file_name=summary_file_path)

        # If the uploaded file is a text file (txt, pdf, etc.)
        elif file_extension in ['txt', 'pdf']:
            # Display the uploaded file content
            st.subheader(f"File Content: {uploaded_file.name}")
            file_content = uploaded_file.read().decode("utf-8")
            st.text_area("File Content", file_content, height=200)

            # Summarize the text content using Gemini API
            if st.button('Summarize the Text'):
                st.spinner('Summarizing the content...')
                gemini_api_key = "AIzaSyDMuyA9vfvAahMCbTSwXFCZ0DsOIbMPWZY"  # Replace with your Gemini API key
                summary_response = summarize_text(gemini_api_key, file_content)

                # Extract summary text from the response
                if isinstance(summary_response, dict):
                    summary_text = summary_response['candidates'][0]['content']['parts'][0]['text']
                else:
                    summary_text = summary_response  # If error occurred

                # Clean the summary text
                cleaned_summary = clean_summary(summary_text)

                # Display the summary
                st.subheader("Summary")
                st.text_area("Summarized Text", cleaned_summary, height=200)

                # Option to save the summary to a file
                summary_file_path = f"summary_{uploaded_file.name}.txt"
                with open(summary_file_path, 'w') as file:
                    file.write(cleaned_summary)
                st.download_button(label="Download Summary", data=open(summary_file_path, 'rb'), file_name=summary_file_path)

if __name__ == "__main__":
    main()
