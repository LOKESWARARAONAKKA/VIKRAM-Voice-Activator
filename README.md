*# VIKRAM – Low Latency and Efficient Voice Activator for Edge Devices*



*## ISRO SIH 2026 – Problem Statement 26172*



*VIKRAM is a low-latency and efficient voice activation system designed for edge devices.*



*The project focuses on detecting the wake word \*\*"VIKRAM"\*\* reliably while keeping computation, memory usage, and response latency suitable for resource-constrained edge hardware.*



*## Project Objective*



*The main objective is to develop a lightweight voice activation system that can:*



*- Continuously listen for the wake word "VIKRAM"*

*- Detect the wake word with low latency*

*- Reduce false activations from normal speech and background sounds*

*- Operate efficiently on edge devices*

*- Work with a compact speech dataset*

*- Provide a foundation for real-time wake-word detection*



*## Wake Word*



*\*\*VIKRAM\*\**



*## Dataset*



*The positive wake-word dataset contains recordings from multiple speakers.*



*Current speakers:*



*- dtss*

*- lavanya*

*- lokesh*

*- mohana*



*The recordings were collected using the following format:*



*- Sample rate: 16 kHz*

*- Channels: Mono*

*- Sample width: 16-bit PCM*

*- Duration: 1.5 seconds*

*- File format: WAV*



*Current positive dataset:*



*- Total recordings: 69*

*- dtss: 18*

*- lavanya: 20*

*- lokesh: 17*

*- mohana: 14*



*## Negative Dataset*



*A negative dataset is also being collected to help the system distinguish the wake word from non-wake-word audio.*



*Examples include:*



*- Normal speech*

*- Different words*

*- Short conversations*

*- Background sounds*

*- Fan/AC noise*

*- Keyboard and desk sounds*

*- Other environmental sounds*



*## Dataset Split*



*The positive dataset has been divided into:*



*- Training set*

*- Validation set*

*- Test set*



*Current split:*



*- Training: 35 recordings*

*- Validation: 20 recordings*

*- Test: 14 recordings*



*## Project Scripts*



*The `scripts` directory contains:*



*### `record\_vikram.py`*



*Records wake-word samples for the VIKRAM dataset.*



*### `record\_negative.py`*



*Records negative samples that do not contain the wake word.*



*### `check\_wav.py`*



*Checks whether WAV recordings follow the required audio format.*



*### `check\_negative\_wav.py`*



*Checks the format and validity of negative audio recordings.*



*### `split\_dataset.py`*



*Splits the collected dataset into training, validation, and test sets.*



*## Current Project Status*



*### Completed*



*- \[x] Git installed and configured*

*- \[x] VIKRAM wake-word recording script*

*- \[x] Positive dataset collection*

*- \[x] Multiple-speaker recordings*

*- \[x] WAV format validation*

*- \[x] Dataset cleaning*

*- \[x] Train/validation/test dataset split*

*- \[x] Negative dataset recording setup*

*- \[ ] Negative dataset collection*

*- \[ ] Feature extraction*

*- \[ ] Model training*

*- \[ ] Model evaluation*

*- \[ ] Model optimization*

*- \[ ] Edge-device deployment*

*- \[ ] Real-time testing*



*## Technologies*



*- Python*

*- WAV / PCM audio*

*- Speech processing*

*- Machine learning*

*- Edge AI*

*- Wake-word detection*



*## Project Structure*



*```text*

*VIKRAM-Voice-Activator/*

*│*

*├── scripts/*

*│   ├── check\_negative\_wav.py*

*│   ├── check\_wav.py*

*│   ├── record\_negative.py*

*│   ├── record\_vikram.py*

*│   └── split\_dataset.py*

*│*

*├── dataset/*

*│*

*├── README.md*

*├── requirements.txt*

*└── .gitignore*

