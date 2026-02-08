import streamlit as st
import os
import tempfile
import cv2
import numpy as np
import plotly.graph_objects as go
from inference import analyze_video, preprocess_video_for_ui, get_frame_importance

# Page Config
st.set_page_config(
    page_title="Deepfake Detection Framework",
    page_icon="🕵️",
    layout="wide"
)

# Sidebar - Instructions
st.sidebar.title("Configuration")
st.sidebar.info("Model: Hybrid ResNet+LSTM")
st.sidebar.info("Modality: Audio + Visual")

st.sidebar.markdown("---")
st.sidebar.header("Instructions")
st.sidebar.markdown("""
1.  **Upload** a video file (MP4/AVI).
2.  Wait for the **Analysis** to complete.
3.  Check the **Fake Probability** gauge.
4.  Explore the **Evidence** tabs to see extracted faces and audio.
5.  View **Frame Analysis** to see which frames contributed most to the decision.
""")

# Title and Description
st.title("🕵️ Deepfake Detection Framework")
st.markdown("""
This application uses a **Hybrid CNN-LSTM-Transformer** model to detect deepfakes in multimedia content.
It analyzes both **Spatial** (Visual) and **Temporal** (Time) features, along with **Audio** spectral patterns.
""")

def create_gauge_chart(probability):
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = probability * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Fake Probability (%)"},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkred" if probability > 0.5 else "green"},
            'steps': [
                {'range': [0, 50], 'color': "lightgreen"},
                {'range': [50, 100], 'color': "lightpink"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 50
            }
        }
    ))
    return fig

def create_bar_chart(scores):
    fig = go.Figure(data=[go.Bar(
        x=[f"Frame {i+1}" for i in range(len(scores))],
        y=scores,
        marker_color='orange'
    )])
    fig.update_layout(
        title="Frame Importance Score (High = Suspicious)",
        xaxis_title="Video Frames",
        yaxis_title="Model Attention",
        yaxis=dict(range=[0, 1])
    )
    return fig

# Main Area
uploaded_file = st.file_uploader("Upload a Video (MP4, AVI)", type=['mp4', 'avi'])

if uploaded_file is not None:
    # Save temp file
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(uploaded_file.read())
    video_path = tfile.name
    tfile.close()

    # Layout
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Original Video")
        st.video(video_path)

    with col2:
        st.subheader("Analysis Results")
        
        with st.spinner("Analyzing Video & Audio..."):
            try:
                # 1. Preprocess
                vid_t, aud_t, faces, spec = preprocess_video_for_ui(video_path)
                
                if vid_t is not None:
                    # Compute Importance
                    importance_scores = get_frame_importance(vid_t, aud_t)
                    
                    # Compute Prediction (reuse tensors)
                    from inference import load_model, DEVICE
                    import torch
                    
                    model = load_model()
                    vid_t = vid_t.to(DEVICE)
                    aud_t = aud_t.to(DEVICE)
                    
                    with torch.no_grad():
                        outputs = model(vid_t, aud_t)
                        probs = torch.softmax(outputs, dim=1)
                        fake_prob = probs[0][1].item()
                    
                    label = "FAKE" if fake_prob > 0.5 else "REAL"
                    
                    # Dashboard Metrics
                    st.plotly_chart(create_gauge_chart(fake_prob), use_container_width=True)
                    
                    if label == "FAKE":
                        st.error(f"## 🚨 PREDICTION: {label}")
                    else:
                        st.success(f"## ✅ PREDICTION: {label}")
                        
                    st.divider()
                    
                    # Explanability
                    st.subheader("Model Evidence")
                    
                    tab1, tab2, tab3 = st.tabs(["Face Crops", "Audio Spectrogram", "Frame Analysis"])
                    
                    with tab1:
                        st.write("Frames extracted for analysis:")
                        if faces:
                            cols = st.columns(5)
                            for i, face_img in enumerate(faces[:10]):
                                with cols[i % 5]:
                                    st.image(face_img, use_container_width=True)
                        else:
                            st.warning("No faces detected.")
                            
                    with tab2:
                        st.write("Mel-Frequency Cepstral Coefficients (MFCCs):")
                        # Normalize for display
                        spec_norm = (spec - spec.min()) / (spec.max() - spec.min())
                        st.image(spec_norm, use_container_width=True)

                    with tab3:
                        st.write("Which frames were most important for the decision? (Gradient Magnitude)")
                        if importance_scores:
                            st.plotly_chart(create_bar_chart(importance_scores), use_container_width=True)
                        else:
                            st.warning("Could not generate frame analysis.")
                        
                else:
                    st.error("Analysis failed. Could not process video.")
                    
            except Exception as e:
                st.error(f"An error occurred: {e}")
                
    # Cleanup
    os.unlink(video_path)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: grey;">
    <b>Deepfake Detection Framework</b> | Built with PyTorch & Streamlit<br>
    <i>This tool is for educational and research purposes only.</i>
</div>
""", unsafe_allow_html=True)
