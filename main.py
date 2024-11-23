import argparse
import os
import socket
import time
from distutils.util import strtobool

import cv2 as cv
import mediapipe as mp
import numpy as np
from dotenv import load_dotenv, find_dotenv

from support import AngleBuffer, estimate_head_pose, get_vector_position, get_blinking_ratio

load_dotenv(find_dotenv())


debug = strtobool(os.getenv("DEBUG", "False"))
enable_head_pose = strtobool(os.getenv("ENABLE_HEAD_POSE", "False"))
show_data = strtobool(os.getenv("SHOW_ON_SCREEN_DATA", "False"))

total_blinks = 0
total_blinks_frame_counter = 0

## Head Pose Estimation Landmark Indices
# These indices correspond to the specific facial landmarks used for head pose estimation.
left_eye_outer_corner = [33]
left_eye_inner_corner = [133]
right_eye_outer_corner = [362]
right_eye_inner_corner = [263]

# Initial Calibration Flags
initial_pitch, initial_yaw, initial_roll = None, None, None
calibrated = False

# Command-line arguments for camera source
parser = argparse.ArgumentParser(description="Eye Tracking Application")
parser.add_argument(
    "-c", "--camSource", help="Source of camera", default=str(os.getenv("DEFAULT_WEBCAM", 0))
)
args = parser.parse_args()


# Face Selected points indices for Head Pose Estimation
_indices_pose = [1, 33, 61, 199, 263, 291]


# Initializing MediaPipe face mesh and camera
if debug:
    print("Initializing the face mesh and camera...")

mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.8,
    min_tracking_confidence=0.8,
)
cam_source = int(args.camSource)
cap = cv.VideoCapture(cam_source)

# Initializing socket for data transmission
iris_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# Main loop for video capture and processing
try:
    angle_buffer = AngleBuffer(size=10)  # Adjust size for smoothing

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Flipping the frame for a mirror effect
        # I think we better not flip to correspond with real world... need to make sure later...
        # frame = cv.flip(frame, 1)
        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        img_h, img_w = frame.shape[:2]
        results = mp_face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            mesh_points = np.array(
                [
                    np.multiply([p.x, p.y], [img_w, img_h]).astype(int)
                    for p in results.multi_face_landmarks[0].landmark
                ]
            )

            # Get the 3D landmarks from facemesh x, y and z(z is distance from 0 points)
            # just normalize values
            mesh_points_3D = np.array(
                [[n.x, n.y, n.z] for n in results.multi_face_landmarks[0].landmark]
            )
            # getting the head pose estimation 3d points
            head_pose_points_3D = np.multiply(
                mesh_points_3D[_indices_pose], [img_w, img_h, 1]
            )
            head_pose_points_2D = mesh_points[_indices_pose]

            # collect nose three dimension and two dimension points
            nose_3D_point = np.multiply(head_pose_points_3D[0], [1, 1, 3000])
            nose_2D_point = head_pose_points_2D[0]

            # create the camera matrix
            focal_length = 1 * img_w

            cam_matrix = np.array(
                [[focal_length, 0, img_h / 2], [0, focal_length, img_w / 2], [0, 0, 1]]
            )

            # The distortion parameters
            dist_matrix = np.zeros((4, 1), dtype=np.float64)

            head_pose_points_2D = np.delete(head_pose_points_3D, 2, axis=1)
            head_pose_points_3D = head_pose_points_3D.astype(np.float64)
            head_pose_points_2D = head_pose_points_2D.astype(np.float64)
            # Solve PnP
            success, rot_vec, trans_vec = cv.solvePnP(
                head_pose_points_3D, head_pose_points_2D, cam_matrix, dist_matrix
            )
            # Get rotational matrix
            rotation_matrix, jac = cv.Rodrigues(rot_vec)

            # Get angles
            angles, mtxR, mtxQ, Qx, Qy, Qz = cv.RQDecomp3x3(rotation_matrix)

            # Get the y rotation degree
            angle_x = angles[0] * 360
            angle_y = angles[1] * 360
            z = angles[2] * 360

            # if angle cross the values then
            threshold_angle = 10
            # See where the user's head tilting
            if angle_y < -threshold_angle:
                face_looks = "Left"
            elif angle_y > threshold_angle:
                face_looks = "Right"
            elif angle_x < -threshold_angle:
                face_looks = "Down"
            elif angle_x > threshold_angle:
                face_looks = "Up"
            else:
                face_looks = "Forward"
            if show_data:
                cv.putText(
                    frame,
                    f"Face Looking at {face_looks}",
                    (img_w - 400, 80),
                    cv.FONT_HERSHEY_TRIPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                    cv.LINE_AA,
                )
            # Display the nose direction
            nose_3d_projection, jacobian = cv.projectPoints(
                nose_3D_point, rot_vec, trans_vec, cam_matrix, dist_matrix
            )

            p1 = nose_2D_point
            p2 = (
                int(nose_2D_point[0] + angle_y * 10),
                int(nose_2D_point[1] - angle_x * 10),
            )

            cv.line(frame, p1, p2, (255, 0, 255), 3)
            eyes_aspect_ratio = get_blinking_ratio(mesh_points_3D)
            if eyes_aspect_ratio <= 0.51: # 0.51 is blink threshold
                total_blinks_frame_counter += 1
            else:
                if total_blinks_frame_counter > 2: # 2 is Number of consecutive frames below the threshold required to confirm a blink.
                    total_blinks += 1
                total_blinks_frame_counter = 0

            # Display all facial landmarks if enabled
            if strtobool(os.getenv("SHOW_FACIAL_LANDMARKS", "False")):
                for point in mesh_points:
                    cv.circle(frame, tuple(point), 1, (0, 255, 0), -1)
            # Process and display eye features
            (l_cx, l_cy), l_radius = cv.minEnclosingCircle(mesh_points[[474, 475, 476, 477]])
            (r_cx, r_cy), r_radius = cv.minEnclosingCircle(mesh_points[[469, 470, 471, 472]])
            center_left = np.array([l_cx, l_cy], dtype=np.int32)
            center_right = np.array([r_cx, r_cy], dtype=np.int32)

            # Highlighting the irises and corners of the eyes
            cv.circle(
                frame, center_left, int(l_radius), (255, 0, 255), 2, cv.LINE_AA
            )  # Left iris
            cv.circle(
                frame, center_right, int(r_radius), (255, 0, 255), 2, cv.LINE_AA
            )  # Right iris
            cv.circle(
                frame, mesh_points[left_eye_inner_corner][0], 3, (255, 255, 255), -1, cv.LINE_AA
            )  # Left eye right corner
            cv.circle(
                frame, mesh_points[left_eye_outer_corner][0], 3, (0, 255, 255), -1, cv.LINE_AA
            )  # Left eye left corner
            cv.circle(
                frame, mesh_points[right_eye_inner_corner][0], 3, (255, 255, 255), -1, cv.LINE_AA
            )  # Right eye right corner
            cv.circle(
                frame, mesh_points[right_eye_outer_corner][0], 3, (0, 255, 255), -1, cv.LINE_AA
            )  # Right eye left corner

            # Calculating relative positions
            l_dx, l_dy = get_vector_position(mesh_points[left_eye_outer_corner], center_left)
            r_dx, r_dy = get_vector_position(mesh_points[right_eye_outer_corner], center_right)

            # Printing data if enabled
            if debug:
                print(f"Total Blinks: {total_blinks}")
                print(f"Left Eye Center X: {l_cx} Y: {l_cy}")
                print(f"Right Eye Center X: {r_cx} Y: {r_cy}")
                print(f"Left Iris Relative Pos Dx: {l_dx} Dy: {l_dy}")
                print(f"Right Iris Relative Pos Dx: {r_dx} Dy: {r_dy}\n")

            if enable_head_pose:
                    pitch, yaw, roll = estimate_head_pose(mesh_points, (img_h, img_w))
                    angle_buffer.add([pitch, yaw, roll])
                    pitch, yaw, roll = angle_buffer.get_average()

                    # Set initial angles on first successful estimation or recalibrate
                    if initial_pitch is None or (key == ord('c') and calibrated):
                        initial_pitch, initial_yaw, initial_roll = pitch, yaw, roll
                        calibrated = True
                        if debug:
                            print("Head pose recalibrated.")

                    # Adjust angles based on initial calibration
                    if calibrated:
                        pitch -= initial_pitch
                        yaw -= initial_yaw
                        roll -= initial_roll

                    if debug:
                        print(f"Head Pose Angles: Pitch={pitch}, Yaw={yaw}, Roll={roll}")
            # Sending data through socket
            timestamp = int(time.time() * 1000)  # Current timestamp in milliseconds
            # Create a packet with mixed types (int64 for timestamp and int32 for the rest)
            packet = np.array([timestamp], dtype=np.int64).tobytes() + np.array([l_cx, l_cy, l_dx, l_dy],
                                                                                dtype=np.int32).tobytes()
            # Writing the on screen data on the frame
            if show_data:
                cv.putText(frame, f"Blinks: {total_blinks}", (30, 80), cv.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 0), 2,
                           cv.LINE_AA)
                if enable_head_pose:
                    cv.putText(frame, f"Pitch: {int(pitch)}", (30, 110), cv.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 0), 2,
                               cv.LINE_AA)
                    cv.putText(frame, f"Yaw: {int(yaw)}", (30, 140), cv.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 0), 2,
                               cv.LINE_AA)
                    cv.putText(frame, f"Roll: {int(roll)}", (30, 170), cv.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 0), 2,
                               cv.LINE_AA)

        # Displaying the processed frame
        cv.imshow("Eye Tracking", frame)
        # Handle key presses
        key = cv.waitKey(1) & 0xFF

        # Calibrate on 'c' key press
        if key == ord('c'):
            initial_pitch, initial_yaw, initial_roll = pitch, yaw, roll
            if debug:
                print("Head pose recalibrated.")

        # Exit on 'q' key press
        if key == ord('q'):
            if debug:
                print("Exiting program...")
            break

except Exception as e:
    print(f"An error occurred: {e}")
finally:
    # Releasing camera and closing windows
    cap.release()
    cv.destroyAllWindows()
    iris_socket.close()
    if debug:
        print("Program exited successfully.")