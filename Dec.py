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
    "Groundnut": "best_9.pt",
    "Chilli": "best.pt",
    "Maize": "best_2.pt",
    "Wheat":"best_14.pt",
    "Sugarcane":"best_4.pt",
    "Paddy":"best_3.pt",
    "Mango":"best_5.pt"
}

# Class lists
CLASS_NAMES = {
    "Groundnut": [
        'aphids','anthracnose','bacterial_wilt',
        'leaf_miner','red_hairy_caterpillar','root_rot','rosette','rust',
        'stunt_virus','tikka','tobacco_caterpillar','white_grub'
    ],
    "Chilli": [
        'aphids','armyworm','caterpillar','fusarium_wilt','mites',
        'powdery_mildew','thirps','whitefly'
    ],
    "Maize":[
        'apids','armyworm','ash_weevil','bacterial_stalk_rot','charcoal_rot','ear_rot',
        'grass_hopper','head_stum','leaf_blight','leaf_hoppers','pink_stem_borer','rajasthan_downy_mildew','rust','shoot_fly'
    ],
    "Wheat":[
        'Aphid', 'Black Rust', 'Blast', 'Rust', 'Common Root Rot',
        'Fusarium Head', 'Leaf Blight', 'Mildew', 'Mite', 'Septoria',
        'Smut', 'Stem_fly', 'Tan spot'
    ],
    "Sugarcane":[
        'aphids','bacteria_blights','downey_mildew','dried_leaves',
        'mealybug','mosaic','red_rot','ring_spot','root_borer','rust',
        'smut','termites','top_borer','yellow_leaf_syndrome','yellow_spot'
    ],
    "Paddy":['Brown spot','False Smut','Leaf Smut','Rice blast',
             'Stem Rot','Tungro','leaf_blight',
             'leaf_folder','sheath_blight'
            ],
    "Mango":['anthracnose','black_rot','fruit_fly','gall_midge',
             'malformation','mango_hopper','mango_mealybug','powdery_mildew',
             'red_rust','sooty_mould','weevil'	
            ]

}

# ----------------------
# UI
# ----------------------
st.title("🌿 Crop Pest & Disease Detection")

crop_choice = st.selectbox("Select Crop", ["Groundnut","Wheat","Maize","Sugarcane","Paddy","Mango","Chilli"])
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











