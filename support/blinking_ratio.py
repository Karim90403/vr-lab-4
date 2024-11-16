from support.euclidean_distance import get_euclidean_distance

right_eye_points = [33, 160, 159, 158, 133, 153, 145, 144]
left_eye_points = [362, 385, 386, 387, 263, 373, 374, 380]


# This function calculates the blinking ratio of a person.
def get_blinking_ratio(landmarks):
    """Calculates the blinking ratio of a person.

    Args:
        landmarks: A facial landmarks in 3D normalized.

    Returns:
        The blinking ratio of the person, between 0 and 1, where 0 is fully open and 1 is fully closed.

    """

    # Get the right eye ratio.
    right_eye_ratio = get_euclidean_distance(landmarks[right_eye_points])

    # Get the left eye ratio.
    left_eye_ratio = get_euclidean_distance(landmarks[left_eye_points])

    # Calculate the blinking ratio.
    ratio = (right_eye_ratio + left_eye_ratio + 1) / 2

    return ratio