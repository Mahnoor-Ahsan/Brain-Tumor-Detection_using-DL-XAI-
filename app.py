import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
from tensorflow.keras.applications.resnet50 import preprocess_input

# 1. Page Configuration
st.set_page_config(page_title="Brain Tumor AI", layout="wide")
st.title("🧠 Brain Tumor Detection with Explainable AI (XAI)")

# 2. Load Your Model (Ensure the .h5 file is in the same folder)
@st.cache_resource
def load_my_model():
    return tf.keras.models.load_model('brain_tumor_model.h5')

model = load_my_model()
labels = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']

# 3. Grad-CAM Logic
def get_gradcam(img_array, model, last_conv_layer_name='conv5_block3_out'):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        class_channel = preds[:, tf.argmax(preds[0])]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    heatmap = last_conv_layer_output[0] @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

# 4. User Interface
uploaded_file = st.file_uploader("Upload an MRI Scan...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Preprocess Image
    image = Image.open(uploaded_file).convert('RGB')
    img = image.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_preprocessed = preprocess_input(np.expand_dims(img_array.copy(), axis=0))

    # Prediction
    preds = model.predict(img_preprocessed)
    class_idx = np.argmax(preds[0])
    confidence = preds[0][class_idx] * 100
    
    # Generate XAI Heatmap
    heatmap = get_gradcam(img_preprocessed, model)
    
    # Display Results
    col1, col2 = st.columns(2)
    with col1:
        st.subheader(f"Prediction: {labels[class_idx]}")
        st.write(f"Confidence: {confidence:.2f}%")
        st.image(image, caption="Uploaded MRI Scan", use_column_width=True)

    with col2:
        st.subheader("AI Focus Area (XAI)")
        # Create overlay
        heatmap_resized = cv2.resize(heatmap, (img.size[0], img.size[1]))
        heatmap_resized = np.uint8(255 * heatmap_resized)
        heatmap_colored = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        
        # Combine original image with heatmap
        superimposed_img = cv2.addWeighted(np.array(img), 0.6, heatmap_colored, 0.4, 0)
        st.image(superimposed_img, caption="Grad-CAM Heatmap", use_column_width=True)
