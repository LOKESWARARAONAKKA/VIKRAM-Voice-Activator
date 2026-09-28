# VIKRAM – Low Latency and Efficient Voice Activator for Edge Devices

## ISRO SIH 2026 – Problem Statement 26172

VIKRAM is a lightweight voice activation system designed for edge devices.

The project focuses on detecting the wake word **"VIKRAM"** using a compact Convolutional Neural Network (CNN), while keeping the model small enough to be suitable for resource-constrained hardware.

## Project Objective

The main objective is to develop a lightweight wake-word detection system that can:

- Detect the wake word "VIKRAM"
- Operate with low inference latency
- Reduce false activations from non-wake-word audio
- Use a compact machine-learning model
- Process speech locally without requiring cloud services
- Provide a foundation for deployment on edge devices

## Wake Word

**VIKRAM**

## Dataset

The positive wake-word dataset contains recordings from multiple speakers.

Current speakers:

- dtss
- lavanya
- lokesh
- mohana
- s.akshitha

### Audio Format

All recordings follow the required format:

- Sample rate: 16 kHz
- Channels: Mono
- Sample width: 16-bit PCM
- Duration: 1.5 seconds
- File format: WAV

### Positive Dataset

Total positive recordings: **84**

| Speaker | Recordings |
|---|---:|
| dtss | 18 |
| lokesh | 17 |
| lavanya | 20 |
| mohana | 14 |
| s.akshitha | 15 |
| **Total** | **84** |

## Negative Dataset

A negative dataset was created to help the model distinguish the wake word from other audio.

Negative samples include:

- Normal speech
- Different English words
- Telugu speech
- Short conversations
- Other people speaking
- Fan and AC noise
- Keyboard sounds
- Desk sounds
- Environmental sounds
- Silence and low-energy audio

Total negative recordings: **60**

## Dataset Split

The main positive dataset was split by speaker to evaluate speaker generalization.

### Training

- dtss: 18
- lokesh: 17
- s.akshitha: 15
- Total positive: 50

### Validation

- lavanya: 20 positive samples
- 10 negative samples
- Total: 30

### Test

- mohana: 14 positive samples
- 10 negative samples
- Total: 24

This speaker-separated structure helps evaluate the model on speakers that were not present in the training set.

## Feature Extraction

Audio is converted into log-Mel spectrogram features.

Configuration:

- Sample rate: 16 kHz
- Mel bands: 40
- FFT size: 512
- Hop length: 160
- Input duration: 1.5 seconds
- Feature shape: approximately `40 × 151`

The same feature-extraction pipeline is used during training and inference.

## CNN Model

A lightweight CNN was developed for wake-word classification.

Architecture:

- Conv2D – 16 filters
- MaxPooling
- Conv2D – 32 filters
- MaxPooling
- Conv2D – 64 filters
- Global Average Pooling
- Dense – 32 neurons
- Dropout – 0.3
- Sigmoid output

### Model Size

Total parameters:

**25,409**

This makes the model relatively compact and suitable as a starting point for edge-device deployment.

## Model Training

The model was trained using:

- Optimizer: Adam
- Learning rate: 0.001 for baseline training
- Loss: Binary Cross-Entropy
- Batch size: 8
- Maximum epochs: 50
- Early stopping based on validation loss

An adapted model was subsequently fine-tuned using controlled recordings to improve performance under the user's recording conditions.

## Model Evaluation

### Original Held-Out Test Set

The baseline model achieved:

**Test accuracy: 91.67%**

The test set contained:

- 14 VIKRAM samples
- 10 negative samples

All 14 VIKRAM samples were correctly detected.

Two negative samples were incorrectly classified as VIKRAM.

### Controlled Recording Evaluation

Additional controlled VIKRAM recordings were collected using the same microphone and recording setup.

The adapted model detected:

**8 / 10 VIKRAM recordings**

Detection rate:

**80%**

These controlled recordings were also used during adaptation, so this result is an adaptation/fit result rather than an independent held-out generalization measurement.

## Real-World Testing

Separate real-world recordings were also evaluated.

The current model showed weaker generalization on these recordings than on the controlled dataset.

Therefore, the current project should **not claim production-level real-world wake-word reliability**.

Further data collection, augmentation, speaker diversity, and independent evaluation are required for robust deployment.

## Live Voice Demo

A real-time microphone inference application was developed.

The live system:

1. Captures 1.5 seconds of microphone audio.
2. Converts the audio into log-Mel spectrogram features.
3. Sends the features to the CNN.
4. Produces a VIKRAM probability.
5. Compares the probability with a detection threshold.
6. Displays whether VIKRAM was detected.

The current live demo uses the computer's microphone and the adapted CNN model.

## Project Scripts

### Dataset Scripts

`record_vikram.py`

Records positive VIKRAM wake-word samples.

`record_negative.py`

Records negative samples that do not contain the wake word.

`check_wav.py`

Validates the required WAV audio format.

`check_negative_wav.py`

Validates negative audio recordings.

`split_dataset.py`

Creates the training, validation, and test dataset structure.

### Machine Learning Scripts

`extract_features_split.py`

Extracts log-Mel spectrogram features from the dataset.

`train_model.py`

Trains the baseline CNN model.

`evaluate_test.py`

Evaluates the model on the held-out test dataset.

`evaluate_speakers.py`

Evaluates VIKRAM predictions across different speakers.

`adapt_model.py`

Fine-tunes the baseline model using controlled recordings.

`test_controlled.py`

Tests the baseline model on controlled recordings.

`test_adapted_controlled.py`

Tests the adapted model on controlled recordings.

`test_adapted_real_world.py`

Evaluates the adapted model on separate real-world recordings.

`live_vikram_demo.py`

Runs real-time microphone-based VIKRAM detection.

## Technologies

- Python
- NumPy
- Librosa
- TensorFlow / Keras
- SoundDevice
- Scikit-learn
- WAV / PCM audio
- Log-Mel spectrograms
- Convolutional Neural Networks
- Edge AI
- Wake-word detection

## Project Structure

```text
VIKRAM-Voice-Activator/
│
├── scripts/
│   ├── check_negative_wav.py
│   ├── check_wav.py
│   ├── record_negative.py
│   ├── record_vikram.py
│   └── split_dataset.py
│
├── training/
│   ├── extract_features_split.py
│   ├── train_model.py
│   ├── evaluate_test.py
│   ├── evaluate_speakers.py
│   ├── adapt_model.py
│   ├── test_controlled.py
│   ├── test_adapted_controlled.py
│   ├── test_adapted_real_world.py
│   ├── live_vikram_demo.py
│   ├── features_train.npz
│   ├── features_validation.npz
│   ├── features_test.npz
│   ├── features_controlled.npz
│   ├── vikram_cnn.keras
│   └── vikram_cnn_adapted.keras
│
├── dataset/
│
├── README.md
├── requirements.txt
└── .gitignore