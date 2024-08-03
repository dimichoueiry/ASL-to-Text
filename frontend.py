import streamlit as st
from hand_tracking import HandTracker

def main():

    st.image('https://cdn.shopify.com/s/files/1/0263/6975/3159/files/6IXTH_SENSE_957ea259-ab1d-4bec-9198-41049ed7bde8.jpg?v=1694717311', width=200)
    st.title('6ixSenseAI - Real-Time ASL to Text Conversion')

    st.write('Welcome to The 6ixth Sense, an app that enables real-time communication for the hard of hearing using Sign Language to text.')
    st.write('With an easy-to-use interface, simply click the START button to convert ASL to text.')

    if 'camera_on' not in st.session_state:
        st.session_state.camera_on = False

    hand_tracker = HandTracker()

    if not st.session_state.camera_on:
        if st.button('START'):
            st.session_state.camera_on = True
            st.rerun()

    if st.session_state.camera_on:
        if st.button('STOP'):
            st.session_state.camera_on = False
            st.rerun()

        st.write('The camera is now on. Please sign in front of the camera to convert ASL to text.')
        stframe = st.empty()

        for frame in hand_tracker.capture_video():
            stframe.image(frame, channels="RGB")

if __name__ == "__main__":
    main()