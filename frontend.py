import streamlit as st
import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Define a function to capture and process video frames
def capture_video():
    cap = cv2.VideoCapture(0)
    hands = mp_hands.Hands()

    stframe = st.empty()

    while st.session_state.camera_on and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Flip the frame horizontally for a later selfie-view display
        frame = cv2.flip(frame, 1)
        
        # Convert the BGR image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame and detect hands
        result = hands.process(rgb_frame)
        
        # Draw hand landmarks
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        
        # Convert the frame to RGB before displaying
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Display the frame in the Streamlit app
        stframe.image(frame, channels="RGB")

    cap.release()

# Streamlit app
def main():
    st.logo('https://th.bing.com/th/id/OIP.3Zz72OvLqlk9nCIKhZSDmwAAAA?pid=ImgDet&w=184&h=184&c=7&dpr=1.3')
    st.image('https://th.bing.com/th/id/OIP.3Zz72OvLqlk9nCIKhZSDmwAAAA?pid=ImgDet&w=184&h=184&c=7&dpr=1.3')

    st.title('6ixSenseAI - Real-Time ASL to Text Conversion')

    st.write('Welcome to The 6ixth Sense, an app that enables real-time communication for the hard of hearing using Sign Language to text.')
    st.write('With an easy-to-use interface, simply click the START button to convert ASL to text.')

    # Initialize session state variables if not already set
    if 'camera_on' not in st.session_state:
        st.session_state.camera_on = False

    if not st.session_state.camera_on:
        if st.button('START'):
            st.session_state.camera_on = True
            st.rerun()

    if st.session_state.camera_on:
        if st.button('STOP'):
            st.session_state.camera_on = False
            st.rerun()

        st.write('The camera is now on. Please sign in front of the camera to convert ASL to text.')
        capture_video()

if __name__ == "__main__":
    main()
