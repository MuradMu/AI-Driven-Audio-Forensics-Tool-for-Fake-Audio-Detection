import os
import numpy as np
import torch
import torch.nn as nn
import librosa
import librosa.display
import matplotlib
matplotlib.use('Agg')  # Use 'Agg' backend for matplotlib to avoid GUI issues
import matplotlib.pyplot as plt
from flask import Flask, request, render_template, redirect, url_for
import soundfile as sf
import gdown

app = Flask(__name__)

# Paths
UPLOAD_FOLDER = 'uploads'
PLOT_FOLDER = 'static/plots'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PLOT_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Download the model from Google Drive
model_url = "https://drive.google.com/uc?id=1-KG-hiydHRgytxWuK5-BDlnY3Djg93ip"  # Replace with your Google Drive file ID
model_path = "audio_forensics_model.pth"
if not os.path.exists(model_path):
    print("Downloading model from Google Drive...")
    gdown.download(model_url, model_path, quiet=False)

# Load the trained model
class AudioCNN(nn.Module):
    def __init__(self):
        super(AudioCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 6 * 50, 128)
        self.fc2 = nn.Linear(128, 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AudioCNN().to(device)
model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
model.eval()

# Parameters
sample_rate = 16000
n_mfcc = 13
n_frames = 100
window_size = 0.5
hop_size = 0.25

# Function to extract MFCCs for a segment
def extract_mfcc_segment(audio, sr, start, window_size, n_mfcc=n_mfcc, n_frames=n_frames):
    start_sample = int(start * sr)
    end_sample = start_sample + int(window_size * sr)
    if end_sample > len(audio):
        end_sample = len(audio)
        audio_segment = audio[start_sample:]
        if len(audio_segment) < int(window_size * sr):
            audio_segment = np.pad(audio_segment, (0, int(window_size * sr) - len(audio_segment)), mode='constant')
    else:
        audio_segment = audio[start_sample:end_sample]
    
    mfccs = librosa.feature.mfcc(y=audio_segment, sr=sr, n_mfcc=n_mfcc)
    if mfccs.shape[1] > n_frames:
        mfccs = mfccs[:, :n_frames]
    else:
        pad_width = n_frames - mfccs.shape[1]
        mfccs = np.pad(mfccs, ((0, 0), (0, pad_width)), mode='constant')
    return mfccs

# Function to extract MFCCs for the entire audio (for overall classification)
def extract_mfcc_full(audio, sr, n_mfcc=n_mfcc, n_frames=n_frames):
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    if mfccs.shape[1] > n_frames:
        mfccs = mfccs[:, :n_frames]
    else:
        pad_width = n_frames - mfccs.shape[1]
        mfccs = np.pad(mfccs, ((0, 0), (0, pad_width)), mode='constant')
    return mfccs

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    
    # Save the uploaded file
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)

    # Load and process the audio
    audio, sr = librosa.load(file_path, sr=sample_rate)
    duration = librosa.get_duration(y=audio, sr=sr)

    # Overall classification
    mfcc = extract_mfcc_full(audio, sr)
    mfcc_tensor = torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(mfcc_tensor)
        probs = torch.softmax(output, dim=1)
        fake_prob = probs[0, 0].item()
        real_prob = probs[0, 1].item()
        prediction = "Fake" if fake_prob > real_prob else "Real"
        confidence = max(fake_prob, real_prob)

    # Generate confidence map
    starts = np.arange(0, duration - window_size + hop_size, hop_size)
    confidence_scores = []
    for start in starts:
        mfcc = extract_mfcc_segment(audio, sr, start, window_size)
        mfcc_tensor = torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(mfcc_tensor)
            probs = torch.softmax(output, dim=1)
            fake_prob = probs[0, 0].item()
            confidence_scores.append(fake_prob)

    # Plot spectrogram with confidence map
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    D = librosa.amplitude_to_db(np.abs(librosa.stft(audio)), ref=np.max)
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title("Spectrogram")

    plt.subplot(2, 1, 2)
    times = starts + window_size / 2
    plt.fill_between(times, confidence_scores, alpha=0.5, color='red', label='Fake Confidence')
    plt.plot(times, confidence_scores, color='red')
    plt.xlabel("Time (s)")
    plt.ylabel("Fake Probability")
    plt.title("Confidence Map (Red = Likely Fake)")
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()

    # Save the plot
    plot_path = os.path.join(PLOT_FOLDER, 'confidence_map.png')
    plt.savefig(plot_path)
    plt.close()

    return render_template('result.html', prediction=prediction, confidence=confidence, plot_url='plots/confidence_map.png')

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)