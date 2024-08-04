import cv2
import mediapipe as mp
import numpy as np

class HandTracker:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.mp_drawing = mp.solutions.drawing_utils

    def __del__(self):
        self.cap.release()

    def capture_video(self):
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            # Flip the frame horizontally for a later selfie-view display
            frame = cv2.flip(frame, 1)
            
            # Convert the BGR image to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process the frame and detect hands
            result = self.hands.process(rgb_frame)
            
            # Draw hand landmarks and extract the data
            saved_hands = []
            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                    hand_landmark_list = [[landmark.x, landmark.y, landmark.z] for landmark in hand_landmarks.landmark]
                    saved_hands.append(hand_landmark_list)
            
            yield frame, saved_hands

