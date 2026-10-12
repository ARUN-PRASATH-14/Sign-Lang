"""
Hand Detector Utility using Google MediaPipe and OpenCV.
Extracts 21 3D hand landmarks, normalizes coordinates, computes bounding boxes,
and handles hand visualization.
"""

import cv2
import numpy as np
import mediapipe as mp

# Multi-tier fallback for MediaPipe solutions across Python/OS versions
try:
    _mp_hands = mp.solutions.hands
    _mp_draw = mp.solutions.drawing_utils
    _mp_draw_styles = mp.solutions.drawing_styles
except (AttributeError, ModuleNotFoundError):
    try:
        import mediapipe.python.solutions.hands as _mp_hands
        import mediapipe.python.solutions.drawing_utils as _mp_draw
        import mediapipe.python.solutions.drawing_styles as _mp_draw_styles
    except (AttributeError, ModuleNotFoundError):
        try:
            from mediapipe import solutions
            _mp_hands = solutions.hands
            _mp_draw = solutions.drawing_utils
            _mp_draw_styles = solutions.drawing_styles
        except Exception:
            _mp_hands = None
            _mp_draw = None
            _mp_draw_styles = None


class HandDetector:
    def __init__(
        self,
        mode: bool = False,
        max_hands: int = 2,
        detection_con: float = 0.7,
        track_con: float = 0.6,
    ):
        self.mode = mode
        self.max_hands = max_hands
        self.detection_con = detection_con
        self.track_con = track_con

        if hasattr(mp, "solutions") and hasattr(mp.solutions, "hands"):
            self.mp_hands = mp.solutions.hands
            self.mp_draw = mp.solutions.drawing_utils
            self.mp_draw_styles = mp.solutions.drawing_styles
        elif _mp_hands is not None:
            self.mp_hands = _mp_hands
            self.mp_draw = _mp_draw
            self.mp_draw_styles = _mp_draw_styles
        else:
            try:
                import mediapipe.python.solutions.hands as h
                import mediapipe.python.solutions.drawing_utils as d
                import mediapipe.python.solutions.drawing_styles as ds
                self.mp_hands = h
                self.mp_draw = d
                self.mp_draw_styles = ds
            except Exception as e:
                raise ImportError(
                    f"MediaPipe solutions not found. Please ensure mediapipe==0.10.14 is installed on Python 3.11/3.12: {e}"
                )

        self.hands = self.mp_hands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.max_hands,
            min_detection_confidence=self.detection_con,
            min_tracking_confidence=self.track_con,
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_draw_styles = mp.solutions.drawing_styles

        # Drawing specs
        self.landmark_style = self.mp_draw.DrawingSpec(
            color=(0, 0, 255), thickness=2, circle_radius=3
        )
        self.connection_style = self.mp_draw.DrawingSpec(
            color=(255, 255, 255), thickness=2, circle_radius=2
        )

        # Tip IDs of fingers: Thumb, Index, Middle, Ring, Pinky
        self.tip_ids = [4, 8, 12, 16, 20]

    def find_hands(self, img: np.ndarray, draw: bool = True):
        """
        Processes image and detects hands.
        Returns img and results object.
        """
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(img_rgb)

        if self.results.multi_hand_landmarks and draw:
            for hand_lms in self.results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(
                    img,
                    hand_lms,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.landmark_style,
                    self.connection_style,
                )
        return img, self.results

    def get_hand_landmarks(self, img: np.ndarray, hand_index: int = 0):
        """
        Extracts pixel landmarks and normalized bounding box for the specified hand.
        Returns:
            lm_list: List of [id, cx, cy, cz] in pixel coordinates
            bbox: (x1, y1, x2, y2)
            handedness: 'Right' or 'Left'
        """
        lm_list = []
        bbox = None
        handedness = "Right"

        if not self.results or not self.results.multi_hand_landmarks:
            return lm_list, bbox, handedness

        if hand_index < len(self.results.multi_hand_landmarks):
            hand_lms = self.results.multi_hand_landmarks[hand_index]
            h, w, c = img.shape
            x_list, y_list = [], []

            for idx, lm in enumerate(hand_lms.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                x_list.append(cx)
                y_list.append(cy)
                lm_list.append([idx, cx, cy, lm.z])

            x1, x2 = min(x_list), max(x_list)
            y1, y2 = min(y_list), max(y_list)
            bbox = (x1, y1, x2, y2)

            # Get handedness
            if (
                self.results.multi_handedness
                and hand_index < len(self.results.multi_handedness)
            ):
                label = self.results.multi_handedness[hand_index].classification[0].label
                # MediaPipe gives mirrored handedness in standard selfie cam
                handedness = label

        return lm_list, bbox, handedness

    def extract_normalized_features(self, hand_lms):
        """
        Normalizes 21 3D coordinates relative to wrist (landmark 0)
        and scales by max span so gesture is position & distance invariant.
        Returns 1D numpy array of length 63: [dx0, dy0, dz0, dx1, dy1, dz1, ...]
        """
        raw_coords = []
        for lm in hand_lms.landmark:
            raw_coords.append([lm.x, lm.y, lm.z])
        raw_coords = np.array(raw_coords)

        # Step 1: Center relative to wrist (landmark 0)
        wrist = raw_coords[0]
        relative_coords = raw_coords - wrist

        # Step 2: Scale normalization by maximum Euclidean distance from wrist
        distances = np.linalg.norm(relative_coords, axis=1)
        max_dist = np.max(distances)
        if max_dist > 1e-6:
            normalized = relative_coords / max_dist
        else:
            normalized = relative_coords

        return normalized.flatten().astype(np.float32)

    def get_features_for_model(self, hand_index: int = 0):
        """
        Returns normalized 63-dim feature vector if hand is found, else None.
        """
        if not self.results or not self.results.multi_hand_landmarks:
            return None
        if hand_index < len(self.results.multi_hand_landmarks):
            hand_lms = self.results.multi_hand_landmarks[hand_index]
            return self.extract_normalized_features(hand_lms)
        return None

    def get_raw_landmarks(self, hand_index: int = 0):
        """
        Returns numpy array of shape (21, 3) with [x, y, z] for each landmark.
        """
        if not self.results or not self.results.multi_hand_landmarks:
            return None
        if hand_index < len(self.results.multi_hand_landmarks):
            hand_lms = self.results.multi_hand_landmarks[hand_index]
            coords = [[lm.x, lm.y, lm.z] for lm in hand_lms.landmark]
            return np.array(coords, dtype=np.float32)
        return None

    def fingers_up(self, lm_list, handedness="Right"):
        """
        Returns list of 5 booleans indicating whether fingers are extended.
        [Thumb, Index, Middle, Ring, Pinky]
        """
        if len(lm_list) < 21:
            return [False] * 5

        fingers = []

        # Thumb: compare x coordinate of tip vs IP joint depending on hand
        if handedness == "Right":
            fingers.append(lm_list[self.tip_ids[0]][1] < lm_list[self.tip_ids[0] - 1][1])
        else:
            fingers.append(lm_list[self.tip_ids[0]][1] > lm_list[self.tip_ids[0] - 1][1])

        # Other 4 fingers: tip y coordinate < PIP joint y coordinate (inverted y axis)
        for i in range(1, 5):
            fingers.append(lm_list[self.tip_ids[i]][2] < lm_list[self.tip_ids[i] - 2][2])

        return fingers

