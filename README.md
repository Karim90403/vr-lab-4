# Work to lab 4 on Virtual Reality In STTANKIN

## Installation & Usage

1. **Clone the Repository:**
   ```
   git clone <link>
   ```

2. **Navigate to the Repository Directory:**
   ```
   cd <dirname>
   ```
3. **Create venv:**
   ```
   python -m venv venv
   ```
   
4. **Activate venv**
   ```
   source venv/bin/activate
   ```

5. **Install Dependencies:**
   ```
   pip install -r requirements.txt
   ```

6. **Run the Application:**
   ```
   python main.py
   ```

   Optionally, specify the camera source:
   ```
   python main.py -c <camera_source_number>
   ```

---
## Interactive Commands

While running the Eye Tracking and Head Pose Estimation script, you can interact with the program using the following keyboard commands:

- **'c' Key**: Calibrate Head Pose
  - Pressing the 'c' key recalibrates the head pose estimation to the current orientation of the user's head. This sets the current head pose as the new reference point.

- **'q' Key**: Quit Program
  - Pressing the 'q' key will exit the program. 


---

## Requirements
- Python 3.x
- OpenCV (opencv-python)
- MediaPipe (mediapipe)
- Other Python standard libraries: `math`, `socket`, `argparse`, `time`, `csv`, `datetime`, `os`

---

## Environment variables
### User-Specific Measurements
USER_FACE_WIDTH: The horizontal distance between the outer edges of the user's cheekbones in millimeters.
This measurement is used to scale the 3D model points for head pose estimation.
Measure your face width and adjust the value accordingly.

```USER_FACE_WIDTH=140``` [mm]

### Configuration Parameters
DEBUG: Enable or disable the printing of data to the console for debugging.

```DEBUG=True```

DEFAULT_WEBCAM: Default camera source index. '0' usually refers to the built-in webcam.

```DEFAULT_WEBCAM = 0```

SHOW_FACIAL_LANDMARKS: If True, display all facial landmarks on the video feed.

```SHOW_FACIAL_LANDMARKS=True```

ENABLE_HEAD_POSE: Enable the head position and orientation estimator.

```ENABLE_HEAD_POSE=True```

### Blink Detection Parameters
SHOW_ON_SCREEN_DATA: If True, display blink count and head pose angles on the video feed.

```SHOW_ON_SCREEN_DATA = False```
