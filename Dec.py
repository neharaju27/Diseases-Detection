import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile
import os
import cv2

# ----------------------
# Load models lazily
# ----------------------
@st.cache_resource
def load_model(model_path):
    return YOLO(model_path)

# Model paths (replace with your files)
MODEL_PATHS = {
    "Groundnut": "best_1.pt",
    "Chilli": "best.pt"
}

# Class lists
CLASS_NAMES = {
    "Groundnut": [
        'aphids','bacterial_wilt','black_fungus_groundnut','bud_rot','leaf_fungus',
        'leaf_miner','red_hairy_caterpillar','root_rot','rosette','rust',
        'stunt_virus','tikka','tobacco_caterpillar','white_grub'
    ],
    "Chilli": [
        'aphids','armyworm','caterpillar','fusarium_wilt','mites',
        'powdery_mildew','thirps','whitefly'
    ]
}

# ----------------------
# UI
# ----------------------
st.title("🌿 Crop Pest & Disease Detection")

crop_choice = st.selectbox("Select Crop", ["Groundnut", "Chilli"])
model = load_model(MODEL_PATHS[crop_choice])
disease_classes = CLASS_NAMES[crop_choice]

st.info(f"ℹ️ **Note:** This model can detect only these classes: {', '.join(disease_classes)}")

uploaded_file = st.file_uploader(f"Upload a {crop_choice} Leaf Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Image (Original)", use_column_width=True)

    # Save temp file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    temp_file.write(uploaded_file.read())
    temp_file.close()

    st.write("🔍 Detecting diseases/pests...")

    # Run YOLO inference
    results = model.predict(source=temp_file.name, conf=0.25, save=False)

    for r in results:
        # Annotated image (BGR)
        im_array = r.plot()
        # Convert to RGB
        im_array = cv2.cvtColor(im_array, cv2.COLOR_BGR2RGB)
        st.image(im_array, caption="Detected Diseases/Pests", use_column_width=True)

        # Extract detected class names
        detected = [disease_classes[int(c)] for c in r.boxes.cls.cpu().numpy()]
        st.subheader("🦠 Detected:")
        if detected:
            st.write(", ".join(set(detected)))
        else:
            st.write("✅ No disease/pest detected!")

    # Cleanup
    os.remove(temp_file.name)