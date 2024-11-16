import os

import cv2 as cv
import numpy as np

from support.normalize_pitch import get_normalized_pitch


def estimate_head_pose(landmarks, image_size):
    # Scale factor based on user's face width (assumes model face width is 150mm)
    scale_factor = int(os.getenv("USER_FACE_WIDTH", 140)) / 150.0
    # 3D model points.
    model_points = np.array([
        (0.0, 0.0, 0.0),  # Nose tip
        (0.0, -330.0 * scale_factor, -65.0 * scale_factor),  # Chin
        (-225.0 * scale_factor, 170.0 * scale_factor, -135.0 * scale_factor),  # Left eye left corner
        (225.0 * scale_factor, 170.0 * scale_factor, -135.0 * scale_factor),  # Right eye right corner
        (-150.0 * scale_factor, -150.0 * scale_factor, -125.0 * scale_factor),  # Left Mouth corner
        (150.0 * scale_factor, -150.0 * scale_factor, -125.0 * scale_factor)  # Right mouth corner
    ])

    # Camera internals
    focal_length = image_size[1]
    center = (image_size[1] / 2, image_size[0] / 2)
    camera_matrix = np.array(
        [[focal_length, 0, center[0]],
         [0, focal_length, center[1]],
         [0, 0, 1]], dtype="double"
    )

    # Assuming no lens distortion
    dist_coeffs = np.zeros((4, 1))

    # 2D image points from landmarks, using defined indices
    image_points = np.array([
        landmarks[4],  # Nose tip
        landmarks[152],  # Chin
        landmarks[33],  # Left eye left corner
        landmarks[263],  # Right eye right corner
        landmarks[61],  # Left mouth corner
        landmarks[291]  # Right mouth corner
    ], dtype="double")

    # Solve for pose
    (success, rotation_vector, translation_vector) = cv.solvePnP(model_points, image_points, camera_matrix, dist_coeffs,
                                                                 flags=cv.SOLVEPNP_ITERATIVE)

    # Convert rotation vector to rotation matrix
    rotation_matrix, _ = cv.Rodrigues(rotation_vector)

    # Combine rotation matrix and translation vector to form a 3x4 projection matrix
    projection_matrix = np.hstack((rotation_matrix, translation_vector.reshape(-1, 1)))

    # Decompose the projection matrix to extract Euler angles
    _, _, _, _, _, _, euler_angles = cv.decomposeProjectionMatrix(projection_matrix)
    pitch, yaw, roll = euler_angles.flatten()[:3]

    # Normalize the pitch angle
    pitch = get_normalized_pitch(pitch)

    return pitch, yaw, roll