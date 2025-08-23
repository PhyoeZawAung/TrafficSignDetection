import streamlit as st

# Page configuration
st.set_page_config(page_title="Traffic Sign Detection", page_icon="🚦", layout="centered")

# Title
st.title("🚦 Traffic Sign Detection")

# Project description
st.markdown("""
Welcome to the **Traffic Sign Detection** project!  

This project uses **YOLOv8** to detect traffic signs in real-time.  
You can upload images or videos, or use a live webcam stream to detect and identify traffic signs.  

The system highlights traffic signs with bounding boxes and labels for easy recognition.
""")
st.page_link("pages/1_Demo.py", label="Go to Demo Page")

