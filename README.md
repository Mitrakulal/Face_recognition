# Face Recognition System

A real-time face recognition system using MediaPipe for fast face detection and face_recognition library for facial feature extraction. The system allows users to enroll new faces by capturing stable face embeddings and recognize enrolled individuals in real-time.

## Features

- **Fast Face Detection**: Uses MediaPipe for efficient real-time face detection
- **Stable Enrollment**: Captures multiple face embeddings only when the face is stable (minimal movement)
- **Real-time Recognition**: Identifies enrolled individuals with confidence threshold
- **Unknown Detection**: Marks unfamiliar faces as "Unknown"
- **Persistent Storage**: Saves face embeddings as `.npy` files for future recognition

## Project Structure

```
Face__recognition/
├── Face_enroll.py          # Enrollment script to register new faces
├── Face_Recognition.py     # Recognition script to identify enrolled faces
├── embeddings/             # Directory storing face embeddings (.npy files)
│   ├── cfggh.npy
│   ├── mitra.npy
│   ├── mk.npy
│   └── pakalaa.npy
└── README.md
```

## Requirements

- Python 3.x
- OpenCV (`cv2`)
- MediaPipe
- face_recognition
- NumPy

## Installation

1. Clone this repository:
```bash
git clone https://github.com/Mitrakulal/Face_recognition.git
cd Face_recognition
```

2. Install required packages:
```bash
pip install opencv-python mediapipe face_recognition numpy
```

## Usage

### Enrolling New Faces

Run the enrollment script:
```bash
python Face_enroll.py
```

**Controls:**
- Press `e` to start enrollment mode
- Enter the person's name when prompted
- Keep your face stable in front of the camera
- The system will capture 10 stable embeddings automatically
- Embeddings are saved in the `embeddings/` folder
- Press `q` to quit

**Enrollment Process:**
- Requires 15 consecutive stable frames before capturing each embedding
- Movement threshold: 10 pixels
- Captures 10 embeddings per person for robust recognition
- Face must remain relatively still during capture

### Recognizing Faces

Run the recognition script:
```bash
python Face_Recognition.py
```

**Controls:**
- The system automatically recognizes enrolled faces in real-time
- Displays the person's name if recognized (distance < 0.4 threshold)
- Shows "Unknown" for unrecognized faces
- Press `q` to quit

## How It Works

### Enrollment (`Face_enroll.py`)
1. Detects face using MediaPipe Face Detection
2. Tracks face stability by monitoring position changes
3. Once stable for 15 frames, captures face encoding using face_recognition
4. Repeats until 10 embeddings are collected
5. Saves embeddings as `.npy` file with person's name

### Recognition (`Face_Recognition.py`)
1. Loads all saved embeddings from `embeddings/` directory
2. Detects faces in real-time using MediaPipe
3. Extracts face encodings using face_recognition
4. Compares with known embeddings using Euclidean distance
5. Identifies person if distance < 0.4 threshold
6. Displays name on video frame

## Configuration

### Enrollment Settings (Face_enroll.py)
- `required_stable = 15`: Frames needed for stability before capture
- `move_thresh = 10`: Maximum pixel movement to consider face stable
- `MAX_em = 10`: Number of embeddings to capture per person

### Recognition Settings (Face_Recognition.py)
- `Threshold = 0.4`: Maximum distance for positive identification
- `min_detection_confidence = 0.5`: MediaPipe detection threshold

## Technical Details

- **Face Detection**: MediaPipe Face Detection (model_selection=1 for full range)
- **Face Encoding**: face_recognition library (dlib-based 128-dimensional embeddings)
- **Matching Algorithm**: Euclidean distance comparison
- **Storage Format**: NumPy arrays saved as `.npy` files

## Troubleshooting

- **Camera not opening**: Check if another application is using the camera
- **No face detected**: Ensure proper lighting and face is clearly visible
- **Recognition not working**: Lower the threshold in Face_Recognition.py or enroll more samples
- **Import errors**: Install all required packages using pip

## Future Improvements

- Add GUI interface
- Support for multiple face recognition in a single frame
- Database integration for larger-scale deployments
- Face anti-spoofing detection
- Confidence score display

## License

This project is open-source and available for educational purposes.

## Author

Mitra Kulal

## Acknowledgments

- MediaPipe by Google for face detection
- face_recognition library by Adam Geitgey
- OpenCV for computer vision operations
