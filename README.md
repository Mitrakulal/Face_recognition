# Face Recognition System

A real-time face recognition system using MediaPipe for fast face detection and face_recognition library for facial feature extraction. The system allows users to enroll new faces by capturing stable face embeddings and recognize enrolled individuals in real-time with temporal smoothing.

## Features

- **Fast Face Detection**: Uses MediaPipe for efficient real-time face detection with model selection 1 (optimized for faces within 5 meters)
- **Stable Enrollment**: Captures multiple face embeddings only when the face is stable (minimal movement)
- **Real-time Recognition**: Identifies enrolled individuals with 0.4 distance threshold
- **Temporal Smoothing**: Uses a 7-frame history with voting mechanism to stabilize recognition results
- **Adaptive Frame Processing**: Processes face encodings every 5 frames to optimize performance
- **Unknown Detection**: Marks unfamiliar faces as "Unknown"
- **Persistent Storage**: Saves face embeddings as `.npy` files for future recognition

## Project Structure

```
lab_vision/
├── Face_enroll.py          # Enrollment script to register new faces
├── Face_Recognition.py     # Recognition script to identify enrolled faces
├── any.py                  # Utility script for MediaPipe path verification
├── embeddings/             # Directory storing face embeddings (.npy files)
│   ├── 085.npy
│   ├── cfggh.npy
│  
└── README.md
```

## Requirements

- **Python 3.x** - Python programming language
  ```bash
  python --version
  ```

- **OpenCV** - Computer vision library
  ```bash
  pip install opencv-python
  ```

- **MediaPipe** - Google's ML framework for face detection
  ```bash
  pip install mediapipe
  ```

- **face_recognition** - Face encoding and recognition library
  ```bash
  pip install face_recognition
  ```

- **NumPy (1.26.4)** - Array processing library (other versions are not compatible with OpenCV)
  ```bash
  pip install numpy==1.26.4
  ```

- Python 3.x
- OpenCV (`cv2`)
- MediaPipe
- face_recognition
- NumPy 1.26.4 (other versions are not compatible with OpenCV)
## Installation

1. Navigate to the project directory:
```bash
cd lab_vision
```

2. Install required packages:
Install required packages with compatible versions:
```
pip install opencv-python
pip install mediapipe
pip install face_recognition
pip install numpy==1.26.4
```

Or install all at once:
```bash
pip install opencv-python mediapipe face_recognition numpy==1.26.4
```

3. Ensure the `embeddings/` directory exists (created automatically during first enrollment)

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

**Recognition Features:**
- **Frame Skipping**: Processes encodings every 5 frames for better performance
- **Temporal Smoothing**: Uses 7-frame history with majority voting to reduce recognition flicker
- **Persistence**: Last known identity is maintained when face temporarily becomes unclear
- **Multiple Embeddings**: Supports comparison against multiple embeddings per person for robust matching

## How It Works

### Enrollment (`Face_enroll.py`)
1. **Face Detection**: Detects face using MediaPipe Face Detection (model selection 1)
2. **Stability Tracking**: Monitors face position to ensure stability (10-pixel movement threshold)
3. **Stable Frame Counting**: Waits for 15 consecutive stable frames before capturing
4. **Encoding Extraction**: Captures face encoding using face_recognition library
5. **Multiple Captures**: Repeats until 10 embeddings are collected per person
6. **Storage**: Saves all embeddings as a single `.npy` file named after the person

**Key Parameters:**
- `required_stable = 15` - Frames needed before capture
- `move_thresh = 10` - Maximum pixel movement for stability
- `MAX_em = 10` - Total embeddings per person
- `min_detection_confidence = 0.5` - MediaPipe detection threshold

### Recognition (`Face_Recognition.py`)
1. **Embedding Loading**: Loads all saved embeddings from `embeddings/` directory
   - Handles both single embeddings and arrays of embeddings
2. **Real-time Detection**: Detects faces using MediaPipe with same parameters as enrollment
3. **Periodic Encoding**: Extracts face encodings every 5 frames (configurable via `Frame_interval`)
4. **Distance Calculation**: Compares current face with all known embeddings using Euclidean distance
5. **Identity Selection**: Chooses the closest match if distance < 0.4 threshold
6. **Temporal Smoothing**: Uses `update_identity()` function with 7-frame history
   - Applies majority voting from recent detections
   - Maintains last known identity when current frame yields no result
7. **Display**: Shows stabilized name on video frame

**Key Parameters:**
- `Threshold = 0.4` - Maximum distance for positive match
- `Frame_interval = 5` - Process every 5th frame
- `hist_size = 7` - Number of frames for temporal smoothing
- `min_detection_confidence = 0.5` - MediaPipe detection threshold

### Temporal Smoothing Algorithm
The recognition system uses a voting-based smoothing mechanism:
```python
def update_identity(current_name):
    # Maintains a history of recent identifications
    # Returns the most common name from last 7 frames
    # Preserves last known identity when face is unclear
```
This prevents flickering between identities and handles temporary detection failures gracefully.

## Configuration

### Enrollment Settings (Face_enroll.py)
- `required_stable = 15`: Frames needed for stability before capture
- `move_thresh = 10`: Maximum pixel movement to consider face stable
- `MAX_em = 10`: Number of embeddings to capture per person
- `min_detection_confidence = 0.5`: MediaPipe detection confidence threshold

### Recognition Settings (Face_Recognition.py)
- `Threshold = 0.4`: Maximum distance for positive identification (lower = stricter)
- `Frame_interval = 5`: Process encodings every N frames (higher = faster but less responsive)
- `hist_size = 7`: Temporal smoothing window size (larger = more stable but slower to update)
- `min_detection_confidence = 0.5`: MediaPipe detection confidence threshold

## Troubleshooting

**Camera not opening:**
- Ensure your webcam is connected and not in use by another application
- Try changing the camera index in `cv2.VideoCapture(0)` to `1` or `2`

**Poor recognition accuracy:**
- Increase number of embeddings (`MAX_em`) during enrollment
- Ensure good lighting conditions during both enrollment and recognition
- Adjust `Threshold` value (lower for stricter matching, higher for looser)

**Recognition flickering:**
- Increase `hist_size` for more temporal smoothing
- Decrease `Frame_interval` for more frequent updates

**Slow performance:**
- Increase `Frame_interval` to process fewer frames
- Use model_selection=0 in MediaPipe for closer faces (< 2 meters)

## Technical Details

- **Face Detection**: MediaPipe FaceDetection with model_selection=1 (optimized for 2-5m range)
- **Face Encoding**: dlib's 128-dimensional face descriptor via face_recognition library
- **Distance Metric**: Euclidean distance between face embeddings
- **Storage Format**: NumPy arrays (.npy) containing 128-d vectors

## Files Description

- **Face_enroll.py**: Interactive enrollment script with stability detection and progress display
- **Face_Recognition.py**: Real-time recognition with temporal smoothing and frame skipping
- **any.py**: Utility script to verify MediaPipe installation path
- **embeddings/**: Directory containing all enrolled face embeddings

## Future Enhancements

- [ ] Support for multiple faces in frame simultaneously
- [ ] GUI interface for easier enrollment and recognition
- [ ] Export/import embeddings database
- [ ] Confidence score display
- [ ] Face quality assessment before enrollment
- [ ] Re-enrollment feature to update existing embeddings

## License

This project is provided as-is for educational purposes.
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
