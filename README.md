# 🎵 Lyrics Cleaner

A Python desktop app that automatically detects and censors explicit words in songs using AI-powered speech recognition.

## How It Works

1. Load an audio file (MP3, WAV, etc.)
2. OpenAI Whisper transcribes the audio with word-level timestamps
3. Explicit words are detected against a built-in word list
4. Those sections are muted/beeped in the audio
5. A clean version is exported

## Requirements

- Python 3.8+
- ffmpeg installed on your system

## Installation

```bash
git clone https://github.com/OwenGY/Lyrics-cleaner.git
cd Lyrics-cleaner
pip install -r requirements.txt
```

### Install ffmpeg

- **Windows**: Download from [ffmpeg.org](https://ffmpeg.org/download.html) and add to PATH
- **Mac**: `brew install ffmpeg`
- **Linux**: `sudo apt install ffmpeg`

## Usage

```bash
python main.py
```

Then use the GUI to load a song and process it.

## Notes

- Larger Whisper models (`medium`, `large`) give better accuracy on fast rap
- Processed files are saved with a `(censored)` suffix so originals are never overwritten