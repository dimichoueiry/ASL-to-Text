import streamlit as st
import torch
from torchvision import models, transforms
import torch.nn.functional as F
from hand_tracking import HandTracker
import numpy as np

class ASLInterpreter:
    def __init__(self):
        self.model = models.mobilenet_v2(pretrained=False)
        num_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(num_features, 26)  # Assuming 26 classes for A-Z
        self.model.load_state_dict(torch.load('asl_model.pth'))
        self.model.eval()
        self.input_size = (224, 224)  # Adjust based on your model's requirements

    def interpret(self, hand_landmarks):
        if not hand_landmarks:
            return "No hands detected"

        # Normalize the hand landmarks
        hand_landmarks = np.array(hand_landmarks)
        hand_landmarks = (hand_landmarks - hand_landmarks.min()) / (hand_landmarks.max() - hand_landmarks.min())

        # Flatten the hand landmarks and convert to tensor
        landmarks_tensor = torch.tensor(hand_landmarks).flatten().unsqueeze(0).float()

        # Resize landmarks tensor to fit model input size
        landmarks_tensor = F.interpolate(landmarks_tensor.unsqueeze(0).unsqueeze(0), size=self.input_size).squeeze(0)

        # Duplicate the single channel to create a 3-channel tensor
        landmarks_tensor = landmarks_tensor.repeat(3, 1, 1)

        # Add batch dimension
        landmarks_tensor = landmarks_tensor.unsqueeze(0)

        with torch.no_grad():
            prediction = self.model(landmarks_tensor)

        # Get the predicted ASL letter
        _, predicted_class = prediction.max(1)
        asl_letter = chr(predicted_class.item() + ord('A'))  # Assuming the model outputs 0-25 for A-Z
        return asl_letter

def main():
    st.image('https://cdn.shopify.com/s/files/1/0263/6975/3159/files/6IXTH_SENSE_957ea259-ab1d-4bec-9198-41049ed7bde8.jpg?v=1694717311', width=200)
    st.title('6ixSenseAI - Real-Time ASL to Text Conversion')

    st.write('Welcome to The 6ixth Sense, an app that enables real-time communication for the hard of hearing using Sign Language to text.')
    st.write('With an easy-to-use interface, simply click the START button to convert ASL to text.')

    if 'camera_on' not in st.session_state:
        st.session_state.camera_on = False

    if 'asl_output' not in st.session_state:
        st.session_state.asl_output = ""

    hand_tracker = HandTracker()
    asl_interpreter = ASLInterpreter()

    if not st.session_state.camera_on:
        if st.button('START'):
            st.session_state.camera_on = True
            st.experimental_rerun()

    if st.session_state.camera_on:
        if st.button('STOP'):
            st.session_state.camera_on = False
            st.experimental_rerun()

        st.write('The camera is now on. Please sign in front of the camera to convert ASL to text.')
        stframe = st.empty()
        text_area = st.empty()  # Placeholder for the text area
        s=0
        while st.session_state.camera_on:
            for frame, saved_hands in hand_tracker.capture_video():
                stframe.image(frame, channels="RGB")
                if saved_hands:
                    asl_output = ""
                    for i, hand_landmarks in enumerate(saved_hands):
                        asl_letter = asl_interpreter.interpret(hand_landmarks)
                        asl_output += f'Hand {i+1}: {asl_letter}\n'
                    st.session_state.asl_output = asl_output

                text_area.text_area('Sign Language Will Appear Here', value=st.session_state.asl_output, height=100, key=s)
                s+=1

if __name__ == "__main__":
    main()
