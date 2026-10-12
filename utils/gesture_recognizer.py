"""
Biomechanical Gesture Recognition Engine.
Calculates anatomical joint angles, finger curl ratios, and Euclidean fingertip proximities
directly from MediaPipe 21 3D landmarks for real-time accuracy across letters and common terms.
"""

import numpy as np


class BiomechanicalGestureRecognizer:
    def __init__(self):
        # MediaPipe landmark indices
        self.WRIST = 0
        self.THUMB_CMC, self.THUMB_MCP, self.THUMB_IP, self.THUMB_TIP = 1, 2, 3, 4
        self.INDEX_MCP, self.INDEX_PIP, self.INDEX_DIP, self.INDEX_TIP = 5, 6, 7, 8
        self.MIDDLE_MCP, self.MIDDLE_PIP, self.MIDDLE_DIP, self.MIDDLE_TIP = 9, 10, 11, 12
        self.RING_MCP, self.RING_PIP, self.RING_DIP, self.RING_TIP = 13, 14, 15, 16
        self.PINKY_MCP, self.PINKY_PIP, self.PINKY_DIP, self.PINKY_TIP = 17, 18, 19, 20

    def _dist(self, p1, p2):
        return float(np.linalg.norm(p1 - p2))

    def analyze_hand(self, raw_landmarks):
        """
        raw_landmarks: numpy array of shape (21, 3) representing [x, y, z] coords.
        Returns dict with biometric metrics and gesture prediction.
        """
        lm = np.array(raw_landmarks)
        if lm.shape[0] < 21:
            return None

        # 1. Palm Scale (Distance from Wrist to Middle MCP)
        palm_scale = self._dist(lm[self.WRIST], lm[self.MIDDLE_MCP])
        if palm_scale < 1e-5:
            palm_scale = 1.0

        # Normalize relative to wrist and palm scale
        coords = (lm - lm[self.WRIST]) / palm_scale

        # 2. Extension ratios for each finger: (Tip to MCP) / (PIP to MCP)
        index_ext_ratio = self._dist(coords[self.INDEX_TIP], coords[self.INDEX_MCP]) / max(
            self._dist(coords[self.INDEX_PIP], coords[self.INDEX_MCP]), 1e-4
        )
        middle_ext_ratio = self._dist(coords[self.MIDDLE_TIP], coords[self.MIDDLE_MCP]) / max(
            self._dist(coords[self.MIDDLE_PIP], coords[self.MIDDLE_MCP]), 1e-4
        )
        ring_ext_ratio = self._dist(coords[self.RING_TIP], coords[self.RING_MCP]) / max(
            self._dist(coords[self.RING_PIP], coords[self.RING_MCP]), 1e-4
        )
        pinky_ext_ratio = self._dist(coords[self.PINKY_TIP], coords[self.PINKY_MCP]) / max(
            self._dist(coords[self.PINKY_PIP], coords[self.PINKY_MCP]), 1e-4
        )

        # Thumb measurements
        thumb_reach = self._dist(coords[self.THUMB_TIP], coords[self.INDEX_MCP])
        thumb_wrist_dist = self._dist(coords[self.THUMB_TIP], coords[self.WRIST])
        thumb_y_tip = coords[self.THUMB_TIP][1]
        thumb_y_mcp = coords[self.THUMB_MCP][1]
        thumb_y_cmc = coords[self.THUMB_CMC][1]

        is_index_ext = index_ext_ratio > 1.35
        is_middle_ext = middle_ext_ratio > 1.35
        is_ring_ext = ring_ext_ratio > 1.35
        is_pinky_ext = pinky_ext_ratio > 1.35
        is_thumb_ext = thumb_reach > 0.65

        # 3. Inter-fingertip distances (normalized)
        d_thumb_index = self._dist(coords[self.THUMB_TIP], coords[self.INDEX_TIP])
        d_thumb_middle = self._dist(coords[self.THUMB_TIP], coords[self.MIDDLE_TIP])
        d_thumb_ring = self._dist(coords[self.THUMB_TIP], coords[self.RING_TIP])
        d_thumb_pinky = self._dist(coords[self.THUMB_TIP], coords[self.PINKY_TIP])
        d_index_middle = self._dist(coords[self.INDEX_TIP], coords[self.MIDDLE_TIP])
        d_middle_ring = self._dist(coords[self.MIDDLE_TIP], coords[self.RING_TIP])
        d_ring_pinky = self._dist(coords[self.RING_TIP], coords[self.PINKY_TIP])

        # Extended finger count
        extended_count = sum([is_index_ext, is_middle_ext, is_ring_ext, is_pinky_ext])

        # 4. Gesture Classification Decision Tree
        pred_label = "Unknown"
        confidence = 0.50

        # --- COMMON TERM: Help / Good (Thumbs Up: fist with thumb pointing straight UP) ---
        if extended_count == 0 and thumb_y_tip < thumb_y_mcp - 0.25 and thumb_reach > 0.50:
            pred_label = "Help / Good"
            confidence = 0.95

        # --- COMMON TERM: Bad (Thumbs Down: fist with thumb pointing straight DOWN) ---
        elif extended_count == 0 and thumb_y_tip > coords[self.WRIST][1] + 0.10:
            pred_label = "Bad"
            confidence = 0.94

        # --- COMMON TERM: Food / Eat (All fingertips bunched together touching thumb tip) ---
        elif (
            d_thumb_index < 0.28
            and d_thumb_middle < 0.28
            and d_thumb_ring < 0.32
            and d_thumb_pinky < 0.38
            and extended_count <= 2
        ):
            pred_label = "Food / Eat"
            confidence = 0.94

        # --- GESTURE: O (Circular ring of thumb and all fingers) ---
        elif d_thumb_index < 0.42 and d_thumb_middle < 0.55 and extended_count <= 2:
            pred_label = "O"
            confidence = 0.94

        # --- COMMON TERM: Okay / F (OK Sign: thumb & index circle, 3 fingers standing UP) ---
        elif d_thumb_index < 0.38 and is_middle_ext and is_ring_ext and is_pinky_ext:
            pred_label = "Okay"
            confidence = 0.96

        # --- COMMON TERM: Pointing / You (Index finger pointing out alone, thumb folded) ---
        elif is_index_ext and not is_middle_ext and not is_ring_ext and not is_pinky_ext and not is_thumb_ext and d_thumb_middle > 0.45:
            pred_label = "You"
            confidence = 0.93

        # --- GESTURE: D (Index extended up, thumb touches middle/ring tips in circle) ---
        elif is_index_ext and not is_middle_ext and not is_ring_ext and not is_pinky_ext and d_thumb_middle < 0.45:
            pred_label = "D"
            confidence = 0.92

        # --- GESTURE: L (Thumb and index at 90 deg L shape) ---
        elif is_index_ext and is_thumb_ext and not is_middle_ext and not is_ring_ext and not is_pinky_ext and d_thumb_index > 0.70:
            pred_label = "L"
            confidence = 0.95

        # --- COMMON TERM: Rock On / Cool (Index and Pinky extended, middle 2 curled, thumb folded) ---
        elif is_index_ext and is_pinky_ext and not is_middle_ext and not is_ring_ext and not is_thumb_ext:
            pred_label = "Rock On"
            confidence = 0.95

        # --- GESTURE / COMMON TERM: I Love You (Thumb, Index, Pinky extended simultaneously) ---
        elif is_index_ext and is_pinky_ext and is_thumb_ext and not is_middle_ext and not is_ring_ext:
            pred_label = "I Love You"
            confidence = 0.96

        # --- COMMON TERM: Call Me / Y (Thumb and Pinky extended outward, middle 3 curled) ---
        elif is_thumb_ext and is_pinky_ext and not is_index_ext and not is_middle_ext and not is_ring_ext:
            pred_label = "Call Me"
            confidence = 0.95

        # --- GESTURE: I (Only Pinky extended) ---
        elif is_pinky_ext and not is_index_ext and not is_middle_ext and not is_ring_ext and not is_thumb_ext:
            pred_label = "I"
            confidence = 0.93

        # --- COMMON TERM: Peace / V (Index and Middle in V shape) ---
        elif is_index_ext and is_middle_ext and not is_ring_ext and not is_pinky_ext:
            if d_index_middle > 0.22:
                pred_label = "Peace"
                confidence = 0.95
            else:
                pred_label = "V"
                confidence = 0.92

        # --- COMMON TERM: Water / W (Index, Middle, Ring extended upright; Pinky curled) ---
        elif is_index_ext and is_middle_ext and is_ring_ext and not is_pinky_ext:
            pred_label = "Water"
            confidence = 0.95

        # --- COMMON TERM: No (Index and Middle pinched onto thumb tip) ---
        elif d_thumb_index < 0.35 and d_thumb_middle < 0.35 and not is_ring_ext and not is_pinky_ext:
            pred_label = "No"
            confidence = 0.93

        # --- COMMON TERM: Stop (Flat open hand with all 5 fingers straight and rigid) ---
        elif extended_count == 4 and is_thumb_ext and d_index_middle < 0.25 and index_ext_ratio > 1.6:
            pred_label = "Stop"
            confidence = 0.93

        # --- COMMON TERM: Hello (All 5 fingers spread out facing camera) ---
        elif extended_count == 4 and is_thumb_ext and d_index_middle >= 0.22:
            pred_label = "Hello"
            confidence = 0.95

        # --- COMMON TERM: Thank You / Please (4 fingers straight together) ---
        elif extended_count == 4 and d_index_middle <= 0.22:
            if d_thumb_middle < 0.50:
                pred_label = "Thank You"
                confidence = 0.92
            else:
                pred_label = "Please"
                confidence = 0.91

        # --- GESTURE: C (Curved hand in C shape) ---
        elif 0.38 < d_thumb_index < 0.85 and 0.95 < index_ext_ratio < 1.45 and 0.95 < middle_ext_ratio < 1.45:
            pred_label = "C"
            confidence = 0.88

        # --- GESTURE: A (Fist with thumb resting beside index) ---
        elif extended_count == 0 and thumb_reach > 0.40 and thumb_y_tip < coords[self.INDEX_MCP][1]:
            pred_label = "A"
            confidence = 0.91

        # --- COMMON TERM: Sorry (Closed fist with thumb across middle) ---
        elif extended_count == 0 and d_thumb_middle < 0.45:
            pred_label = "Sorry"
            confidence = 0.90

        # --- COMMON TERM: Yes (Closed fist) ---
        elif extended_count == 0:
            pred_label = "Yes"
            confidence = 0.90

        # --- GESTURE: E (Fingers curled tightly, thumb tucked under) ---
        elif extended_count == 0 and thumb_y_tip > coords[self.INDEX_DIP][1]:
            pred_label = "E"
            confidence = 0.88

        else:
            # Fallback based on closest feature configuration
            if extended_count >= 3:
                pred_label = "Hello"
                confidence = 0.70
            elif extended_count == 0:
                pred_label = "Yes"
                confidence = 0.70
            elif is_index_ext:
                pred_label = "You"
                confidence = 0.70

        return {
            "label": pred_label,
            "confidence": confidence,
            "extended_count": extended_count,
            "is_thumb_ext": is_thumb_ext,
            "is_index_ext": is_index_ext,
            "is_middle_ext": is_middle_ext,
            "is_ring_ext": is_ring_ext,
            "is_pinky_ext": is_pinky_ext,
        }
