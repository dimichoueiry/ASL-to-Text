import streamlit as st


st.logo('https://th.bing.com/th/id/OIP.3Zz72OvLqlk9nCIKhZSDmwAAAA?pid=ImgDet&w=184&h=184&c=7&dpr=1.3')
st.image('https://th.bing.com/th/id/OIP.3Zz72OvLqlk9nCIKhZSDmwAAAA?pid=ImgDet&w=184&h=184&c=7&dpr=1.3')

st.title('6ixSenseAI - Real-Time ASL to Text Conversion')


st.write('Welcome to The 6ixth Sense, an app that enables real-time communication for the hard of hearing using Sign Language to text.')
st.write('With an easy-to-use interface, simply click the START button to convert ASL to text.')


start_button = st.button('START')

if(start_button):
    st.write('The camera is now on. Please sign in front of the camera to convert ASL to text.')
    st.camera_input('Camera Input', )
    st.text_area('Sign Language Will Appear Here') 

    stop_button = st.button('STOP')

    if(stop_button):
        start_button = False



