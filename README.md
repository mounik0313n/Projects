# Audio Transcription and Summarization Web Application

This project provides a web application built with **Streamlit** that allows users to upload audio or text-based files for transcription and summarization. The app utilizes the **AssemblyAI API** for transcribing audio files and the **Gemini API** for generating summaries of the transcribed text or any text content.

## Features

- **Audio File Transcription**: Upload an audio file (MP3, WAV, or M4A), and the app will transcribe the speech into text using **AssemblyAI API**.
- **Text File Summarization**: Upload a text file (TXT or PDF), and the app will summarize the text content using **Gemini API**.
- **Summary Download**: After generating the summary, users can download the summarized text as a `.txt` file.
- **File Previews**: The app displays the content of audio files (before transcription) and text files (before summarization).

## Technologies Used

- **Streamlit**: A framework for building interactive web applications in Python.
- **AssemblyAI API**: Provides speech-to-text transcription for various audio file formats.
- **Gemini API**: Used for summarizing text content from transcription or directly uploaded text files.

## Code Explanation

### 1. **Imports**

```python
import streamlit as st
import assemblyai as aai
import time
import sys
import requests
import json
import os
