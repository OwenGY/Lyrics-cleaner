import os
import re
import time
import threading
import numpy as np
from flask import Flask, request, jsonify, render_template, send_file
from werkzeug.utils import secure_filename
import whisper
from pydub import AudioSegment
from mutagen import File as MutagenFile
from mutagen.id3 import ID3, ID3NoHeaderError
from mutagen.mp3 import MP3

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
ALLOWED_EXTENSIONS = {'mp3', 'wav', 'm4a', 'ogg', 'flac'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

job_status = {
    'status': 'idle',
    'message': '',
    'found_count': 0,
    'transcript': '',
    'censored_words': [],
    'output_file': None,
}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def generate_beep(duration_ms, volume_rms):
    """Generate a sine wave beep matched to the surrounding audio volume."""
    sample_rate = 44100
    num_samples = int(sample_rate * duration_ms / 1000)
    t = np.linspace(0, duration_ms / 1000, num_samples, endpoint=False)
    sine = np.sin(2 * np.pi * 1000 * t)

    # Match volume to surrounding audio
    target_amplitude = max(volume_rms, 500)
    sine_rms = np.sqrt(np.mean(sine ** 2))
    if sine_rms > 0:
        sine = sine * (target_amplitude / sine_rms / 32768)

    samples = (sine * 32767).astype(np.int16)
    beep = AudioSegment(
        samples.tobytes(),
        frame_rate=sample_rate,
        sample_width=2,
        channels=1,
    )
    return beep.set_channels(2)  # stereo to match typical MP3


def clean_word(w):
    """Strip punctuation from a word for comparison."""
    return re.sub(r"[^a-z0-9']", '', w.lower())


def word_matches(word, censor_set, match_mode):
    """
    Check if a transcribed word should be censored based on match mode.
    - exact:      word == censor word
    - contains:   censor word appears anywhere inside the word
    - startswith: word begins with the censor word
    """
    if match_mode == 'exact':
        return word in censor_set
    elif match_mode == 'contains':
        return any(censor in word for censor in censor_set)
    elif match_mode == 'startswith':
        return any(word.startswith(censor) for censor in censor_set)
    return word in censor_set


def process_audio(file_path, censor_words, model_size, match_mode):
    global job_status

    try:
        # ── Step 1: Transcribe ──────────────────────────────────────
        job_status.update({'status': 'transcribing', 'message': 'Loading Whisper model...', 'found_count': 0})
        model = whisper.load_model(model_size)

        job_status['message'] = 'Transcribing audio...'
        result = model.transcribe(file_path, word_timestamps=True)

        # ── Step 2: Find censored word timestamps ───────────────────
        job_status['message'] = 'Analysing transcript...'
        censor_set = {clean_word(w) for w in censor_words if w.strip()}
        hits = []  # list of (start_ms, end_ms, matched_word)

        for seg in result.get('segments', []):
            for word_info in seg.get('words', []):
                word = clean_word(word_info.get('word', ''))
                if word_matches(word, censor_set, match_mode):
                    start_ms = int(word_info['start'] * 1000)
                    end_ms = int(word_info['end'] * 1000)
                    hits.append((max(0, start_ms - 30), end_ms + 30, word))

        matched_words = list({h[2] for h in hits})  # unique matched words
        job_status['found_count'] = len(hits)
        job_status['message'] = f'Found {len(hits)} instance(s) — processing audio...'

        # ── Step 3: Replace each hit with a beep ────────────────────
        audio = AudioSegment.from_file(file_path)

        # Process in reverse order so timestamps stay valid
        for start_ms, end_ms, _ in sorted(hits, reverse=True):
            end_ms = min(end_ms, len(audio))
            duration = end_ms - start_ms
            if duration <= 0:
                continue
            volume = audio[start_ms:end_ms].rms
            beep = generate_beep(duration, volume)
            audio = audio[:start_ms] + beep + audio[end_ms:]

        # ── Step 4: Export ───────────────────────────────────────────
        job_status['message'] = 'Exporting file...'
        original_basename = os.path.splitext(os.path.basename(file_path))[0]
        output_filename = f'{original_basename} (censored).mp3'
        output_path = os.path.join(OUTPUT_FOLDER, output_filename)
        audio.export(output_path, format='mp3', bitrate='192k')

        # ── Step 5: Copy original metadata to output file ────────────
        try:
            # Read original title before we do anything else
            original_title = None
            try:
                src_easy = MutagenFile(file_path, easy=True)
                if src_easy and 'title' in src_easy:
                    original_title = src_easy['title']  # list of title values
            except Exception:
                pass

            # Copy all easy tags (title, artist, album, year, genre, etc.)
            original_tags = MutagenFile(file_path, easy=True)
            if original_tags is not None:
                output_tags = MutagenFile(output_path, easy=True)
                if output_tags is not None:
                    for key, value in original_tags.items():
                        try:
                            output_tags[key] = value
                        except Exception:
                            pass
                    # Explicitly restore title in case it was overwritten
                    if original_title:
                        output_tags['title'] = original_title
                    output_tags.save()

            # Copy full ID3 tags (album art, lyrics, etc.) for MP3 files
            try:
                src_id3 = ID3(file_path)
                dst_id3 = ID3(output_path)
                for key, value in src_id3.items():
                    dst_id3[key] = value
                # Explicitly preserve TIT2 (title frame) from source
                if 'TIT2' in src_id3:
                    dst_id3['TIT2'] = src_id3['TIT2']
                dst_id3.save(output_path)
            except (ID3NoHeaderError, Exception):
                pass

        except Exception:
            pass  # Metadata copy failing should never block the download

        # ── Step 6: Build highlighted transcript ─────────────────────
        raw_transcript = result.get('text', '')

        job_status.update({
            'status': 'complete',
            'message': f'Done — {len(hits)} word(s) censored. Ready to download.',
            'transcript': raw_transcript,
            'censored_words': list(censor_set),
            'matched_words': matched_words,
            'match_mode': match_mode,
            'output_file': output_path,
        })

    except Exception as e:
        job_status.update({
            'status': 'error',
            'message': f'Error: {str(e)}',
        })


# ── Routes ────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    global job_status

    if job_status['status'] in ('transcribing', 'processing'):
        return jsonify({'error': 'A job is already running'}), 400

    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if not file or file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'Unsupported file type'}), 400

    censor_words = request.form.get('censorWords', '').split(',')
    model_size = request.form.get('modelSize', 'base')
    match_mode = request.form.get('matchMode', 'contains')

    if model_size not in ('tiny', 'base', 'small'):
        model_size = 'base'
    if match_mode not in ('exact', 'contains', 'startswith'):
        match_mode = 'contains'

    filename = secure_filename(file.filename)
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(file_path)

    # Reset state
    job_status.update({
        'status': 'starting',
        'message': 'Starting...',
        'found_count': 0,
        'transcript': '',
        'censored_words': [],
        'output_file': None,
    })

    t = threading.Thread(
        target=process_audio,
        args=(file_path, censor_words, model_size, match_mode),
        daemon=True,
    )
    t.start()

    return jsonify({'message': 'Upload received, processing started'})


@app.route('/status')
def get_status():
    return jsonify(job_status)


@app.route('/download')
def download_file():
    output_path = job_status.get('output_file')
    if not output_path or not os.path.exists(output_path):
        return jsonify({'error': 'No processed file available'}), 404
    return send_file(output_path, as_attachment=True, download_name='censored_audio.mp3')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=False, threaded=True)