# AI-Driven Audio Forensics Tool for Fake Audio Detection

## Model Download:
https://drive.google.com/file/d/1-KG-hiydHRgytxWuK5-BDlnY3Djg93ip/view?usp=sharing

## Overview
This project is an **AI-driven audio forensics tool** designed to detect whether an audio clip is real (human-recorded) or fake (AI-generated, e.g., deepfake). It uses signal processing (MFCC features) and a convolutional neural network (CNN) to classify audio with 93% accuracy. A unique feature of this tool is the **confidence map**, which visualizes which segments of the audio are likely fake, making it interpretable and practical for forensics applications.

The tool is deployed as a Flask web app, allowing users to upload audio files, get a real/fake prediction, and view the confidence map—all in a user-friendly interface with a modern dark-themed design.

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
- **Heroku**: Hosting the deployed app.
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

Heroku for free hosting.

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
