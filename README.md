# AI-Driven Audio Forensics Tool for Fake Audio Detection

To test the model hosted on Render:

https://fake-audio-detection-nf9l.onrender.com/

## Model Download:
https://drive.google.com/file/d/1-KG-hiydHRgytxWuK5-BDlnY3Djg93ip/view?usp=sharing

![Screenshot 2025-03-30 224540](https://github.com/user-attachments/assets/4cb1ff7e-7d22-472a-8c1d-1ced434b1ba7)
![Screenshot 2025-03-30 224609](https://github.com/user-attachments/assets/146a6d62-ab53-4341-b30b-bbb4f174772e)
![Screenshot 2025-03-30 224552](https://github.com/user-attachments/assets/d2ee89cc-0ea8-4346-a143-c2e35f028eed)
![Screenshot 2025-03-30 224649](https://github.com/user-attachments/assets/ab49efa0-7d1d-4cb5-814a-e22004d5397e)

## Overview
This project is an **AI-driven audio forensics tool** designed to detect whether an audio clip is real (human-recorded) or fake (AI-generated, e.g., deepfake). It uses signal processing (MFCC features) and a convolutional neural network (CNN) to classify audio with 93% accuracy. A unique feature of this tool is the **confidence map**, which visualizes which segments of the audio are likely fake, making it interpretable and practical for forensics applications.

The tool is deployed as a Flask web app, allowing users to upload audio files, get a real/fake prediction, and view the confidence map—all in a user-friendly interface with a modern dark-themed design.

## Instructions for Running Locally

To run the Audio Forensics Tool locally on your machine, follow these steps:

### Clone the Repository:
Clone the GitHub repository to your local machine:

```bash
git clone https://github.com/MuradMu/Fake-Audio-Detection.git
cd Fake-Audio-Detection
```

### Set Up a Virtual Environment (Optional but Recommended):

Create a virtual environment to isolate dependencies:

```
python -m venv venv
```

Activate the virtual environment:
On Windows:

```
.\venv\Scripts\activate
```

On macOS/Linux:

```
source venv/bin/activate
```

### Install Dependencies:

Install the required Python packages listed in requirements.txt.

```
pip install -r requirements.txt
```

This installs Flask, PyTorch, Librosa, and other dependencies.

### Download the Pre-Trained Model (optional):

The pre-trained model (audio_forensics_model.pth) is hosted on Google Drive due to GitHub’s file size limits.

Update the model path in main.py with the path to the local file:

```
model_url = "https://drive.google.com/uc?id=1-KG-hiydHRgytxWuK5-BDlnY3Djg93ip"
```

If not done the app will automatically download the model at runtime using gdown. 

### Run the App:

Start the Flask app:

```
python main.py
```

The app will download the model (if not already present) and start the server.

Access the App:

Open your browser and go to http://127.0.0.1:5000.

You should see the upload page. Upload a WAV audio file (e.g., a small file under 5MB, such as a 2-second clip) to classify it as real or fake and view the confidence map.

## Features
- **Audio Classification**: Classifies audio as real or fake with 93% accuracy using a CNN trained on the [Fake-or-Real Dataset](https://www.kaggle.com/datasets/mohammedabdeldayem/the-fake-or-real-dataset).
- **Confidence Map**: Visualizes the probability of fakery over time, highlighting suspicious segments.
- **Web Interface**: Built with Flask, featuring a modern dark-themed UI with custom CSS styling.
- **Scalable Deployment**: Deployed on Heroku for online access.

## Tech Stack
- **Python**: Core language for the project.
- **Librosa**: For audio signal processing (MFCC feature extraction).
- **PyTorch**: For building and training the CNN.
- **Flask**: Web framework for the user interface.
- **CSS**: Custom styling for a modern, user-friendly design.
- **Google Colab**: Used for training the model on GPU.

## Dataset
The model was trained on a subset of the [Fake-or-Real Dataset](https://www.kaggle.com/datasets/mohammedabdeldayem/the-fake-or-real-dataset) from Kaggle, containing 10,208 audio clips (5104 fake, 5104 real). 

## Results
Model Accuracy: 93.19% on the test set.

Precision: 92.49%

Recall: 94.03%

F1 Score: 93.25%

## License
This project is licensed under the MIT License—see the  file for details.

Acknowledgments

Fake-or-Real Dataset for providing the training data.

Librosa and PyTorch for their amazing libraries.

MIT License

Copyright (c) 2025 Murad

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
