import cv2
import mediapipe as mp
import os
class HandTracker:
    def __init__(self, save_dir='saved_hands'):
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.hand_count = 0

    def process_frame(self, frame):
        hands = self.mp_hands.Hands()
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb_frame)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                self._save_hand_image(frame, hand_landmarks)
                self.mp_drawing.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

        return rgb_frame  # Return the RGB frame

    def _save_hand_image(self, frame, hand_landmarks):
        h, w, _ = frame.shape
        hand_bbox = [
            min(lm.x for lm in hand_landmarks.landmark), 
            min(lm.y for lm in hand_landmarks.landmark), 
            max(lm.x for lm in hand_landmarks.landmark), 
            max(lm.y for lm in hand_landmarks.landmark)
        ]
        x_min = int(hand_bbox[0] * w)
        y_min = int(hand_bbox[1] * h)
        x_max = int(hand_bbox[2] * w)
        y_max = int(hand_bbox[3] * h)

        padding = 20
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(w, x_max + padding)
        y_max = min(h, y_max + padding)

        hand_image = frame[y_min:y_max, x_min:x_max]
        hand_image_path = os.path.join(self.save_dir, f'hand_{self.hand_count}.jpg')
        cv2.imwrite(hand_image_path, cv2.cvtColor(hand_image, cv2.COLOR_RGB2BGR))
        self.hand_count += 1

    def capture_video(self):
        cap = cv2.VideoCapture(0)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            processed_frame = self.process_frame(frame)
            yield processed_frame  # Yield the RGB frame
        cap.release()