import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, classification_report
import os

# ============================================================
# PLANTVISION AI - FINAL STREAMLIT APPLICATION
# ============================================================

st.set_page_config(
    page_title="PlantVision AI",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------------- CSS -------------------------------

st.markdown("""
<style>
.stApp {
    background: #f5f8f4;
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg,#123524 0%,#1d5a3a 55%,#123524 100%);
    border-right: 1px solid #2f704d;
}
[data-testid="stSidebar"] * {
    color: #f5efe8 !important;
}
[data-testid="stSidebar"] .stButton > button {
    width: 100%;
    min-height: 52px;
    margin: 5px 0;
    border: 1px solid #2f704d;
    border-radius: 14px;
    background: rgba(255,255,255,.08);
    color: #f5efe8 !important;
    font-weight: 650;
    text-align: left;
    padding: 0 16px;
    transition: all .2s ease;
    box-shadow: 0 5px 18px rgba(0,0,0,.10);
}
[data-testid="stSidebar"] .stButton > button:hover {
    background: linear-gradient(90deg,#24583c,#2f704d);
    border-color: #62b27d;
    transform: translateX(3px);
    box-shadow: 0 8px 22px rgba(0,0,0,.18);
}
[data-testid="stSidebar"] .stButton > button:focus {
    border-color: #7bc895;
    box-shadow: 0 0 0 2px rgba(123,200,149,.22);
}
[data-testid="stSidebar"] hr {
    border-color: #2f704d;
}
.brand {
    padding: 4px 2px 18px 2px;
}
.logo-wrap {
    display:flex;
    align-items:center;
    gap:12px;
}
.logo-mark {
    width:50px;
    height:50px;
    border-radius:15px;
    display:flex;
    align-items:center;
    justify-content:center;
    background:linear-gradient(135deg,#2f8f5b,#1f6b43);
    box-shadow:0 10px 25px rgba(47,143,91,.35);
    font-size:27px;
}
.brand h1 {
    margin: 0;
    font-size: 23px;
    font-weight: 800;
    letter-spacing: -.4px;
}
.brand p {
    margin: 3px 0 0 0;
    color:#b7d2bf !important;
    opacity:1;
    font-size:11px;
}
.sidebar-label {
    margin: 4px 0 8px 3px;
    font-size:10px;
    font-weight:800;
    letter-spacing:1.5px;
    color:#9fc0aa !important;
    text-transform:uppercase;
}
.sidebar-panel {
    padding:13px;
    border:1px solid #2b3b55;
    border-radius:15px;
    background:linear-gradient(145deg,rgba(255,255,255,.10),rgba(255,255,255,.045));
    margin-top:12px;
}
.sidebar-panel .row {
    display:flex;
    justify-content:space-between;
    margin:6px 0;
    font-size:11px;
    color:#b4cdbb !important;
}
.sidebar-panel .value {
    color:#eef7f0 !important;
    font-weight:700;
}
.live-dot {
    display:inline-block;
    width:8px;
    height:8px;
    border-radius:50%;
    background:#4ade80;
    box-shadow:0 0 10px #4ade80;
    margin-right:6px;
}
.nav-note {
    color:#9fc0aa !important;
    font-size:10px;
    margin:8px 2px 0;
    line-height:1.5;
}
.hero {
    padding: 34px 38px;
    border-radius: 24px;
    background: linear-gradient(135deg,#123524,#28734a);
    color: white;
    margin-bottom: 24px;
}
.hero h1 {
    font-size: 42px;
    margin: 0 0 8px 0;
}
.hero p {
    font-size: 17px;
    margin: 0;
    opacity: .9;
}
.tag {
    display:inline-block;
    padding:6px 11px;
    margin:14px 6px 0 0;
    border-radius:20px;
    background:rgba(255,255,255,.14);
    font-size:12px;
}
.card {
    background:white;
    border-radius:18px;
    padding:22px;
    border:1px solid #e4ebe5;
    box-shadow:0 5px 20px rgba(20,60,35,.06);
    margin-bottom:18px;
}
.metric {
    background:white;
    border:1px solid #e4ebe5;
    border-radius:16px;
    padding:18px;
    text-align:center;
    min-height:148px;
    display:flex;
    flex-direction:column;
    align-items:center;
    justify-content:flex-start;
}
.metric .value {
    font-size:27px;
    font-weight:800;
    height:42px;
    min-height:42px;
    display:flex;
    align-items:center;
    justify-content:center;
    width:100%;
}
.metric .label {
    color:#64756a;
    font-size:13px;
    height:20px;
    min-height:20px;
    display:flex;
    align-items:center;
    justify-content:center;
    width:100%;
}
.metric h3 {
    margin:8px 0 0 0 !important;
    min-height:24px;
    display:flex;
    align-items:center;
    justify-content:center;
}
.section-title {
    font-size:28px;
    font-weight:800;
    color:#173c29;
    margin:10px 0 5px 0;
}
.muted {
    color:#66756b;
}
.result {
    border-radius:18px;
    padding:24px;
    background:#edf8ef;
    border:1px solid #cfe7d3;
}
.info {
    border-radius:16px;
    padding:18px;
    background:#f1f6f2;
    border:1px solid #dce7de;
}
.disease {
    min-height:150px;
}
.small {
    font-size:12px;
    color:#6c7a70 !important;
}

/* =========================================================
   GLOBAL TEXT VISIBILITY
   ========================================================= */
.card, .card *, .metric, .metric *, .result, .result *, .info, .info *, .disease, .disease * {
    color: #26352d !important;
}

.card h1, .card h2, .card h3, .card h4,
.metric h1, .metric h2, .metric h3, .metric h4,
.result h1, .result h2, .result h3, .result h4,
.info h1, .info h2, .info h3, .info h4 {
    color: #173c29 !important;
}

.hero, .hero * {
    color: #ffffff !important;
}

[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li,
[data-testid="stMarkdownContainer"] label {
    color: #26352d !important;
}

[data-testid="stFileUploader"] label,
[data-testid="stCameraInput"] label,
[data-testid="stRadio"] label,
[data-testid="stWidgetLabel"] p {
    color: #26352d !important;
}

[data-testid="stMetricLabel"],
[data-testid="stMetricValue"],
[data-testid="stMetricDelta"] {
    color: #26352d !important;
}

[data-testid="stDownloadButton"] button {
    background: #1f6b43 !important;
    color: #ffffff !important;
    border: 1px solid #1f6b43 !important;
    font-weight: 700 !important;
}

[data-testid="stDownloadButton"] button * {
    color: #ffffff !important;
}

/* =========================================================
   MOBILE RESPONSIVE FIX
   ========================================================= */
@media (max-width: 768px) {
    .hero {
        padding: 24px 20px !important;
        border-radius: 18px !important;
    }

    .hero h1 {
        font-size: 30px !important;
        line-height: 1.15 !important;
    }

    .hero p {
        font-size: 14px !important;
        line-height: 1.5 !important;
    }

    .tag {
        font-size: 11px !important;
        margin-top: 10px !important;
    }

    .card, .metric, .result, .info {
        padding: 16px !important;
        color: #26352d !important;
    }

    .card *, .metric *, .result *, .info *, .disease * {
        color: #26352d !important;
    }

    .hero * {
        color: #ffffff !important;
    }

    .section-title {
        font-size: 23px !important;
        color: #173c29 !important;
    }

    .metric {
        min-height: 125px !important;
        margin-bottom: 10px !important;
    }

    .metric .value {
        font-size: 23px !important;
    }

    .metric .label {
        font-size: 12px !important;
    }

    .disease {
        min-height: auto !important;
    }

    [data-testid="stHorizontalBlock"] {
        gap: 0.75rem !important;
    }

    [data-testid="stImage"] img {
        max-width: 100% !important;
        height: auto !important;
    }

    [data-testid="stFileUploaderDropzone"] {
        padding: 12px !important;
    }

    .stButton > button,
    [data-testid="stDownloadButton"] button {
        min-height: 46px !important;
    }

    .sidebar-panel .row {
        font-size: 10px !important;
    }
}

/* Never let custom HTML create a horizontal scroll on small screens. */
html, body, [data-testid="stAppViewContainer"] {
    overflow-x: hidden !important;
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# REAL TRAINING RESULTS FROM THE UPLOADED COLAB NOTEBOOK
# ============================================================

EPOCHS = [1, 2, 3, 4, 5, 6, 7]
TRAIN_ACC = [0.7883, 0.9131, 0.9336, 0.9364, 0.9408, 0.9531, 0.9450]
VAL_ACC = [0.9067, 0.9333, 0.9378, 0.9356, 0.9489, 0.9511, 0.9433]
TRAIN_LOSS = [0.5118, 0.2474, 0.1966, 0.1761, 0.1592, 0.1428, 0.1494]
VAL_LOSS = [0.2598, 0.2011, 0.1846, 0.1592, 0.1456, 0.1468, 0.1466]

BEST_VAL_ACC = max(VAL_ACC)
BEST_EPOCH = EPOCHS[int(np.argmax(VAL_ACC))]

class_names = [
    "Tomato_Early_blight",
    "Tomato_Late_blight",
    "Tomato_healthy"
]

display_names = {
    "Tomato_Early_blight": "Tomato Early Blight",
    "Tomato_Late_blight": "Tomato Late Blight",
    "Tomato_healthy": "Healthy Tomato Leaf"
}

disease_info = {
    "Tomato_Early_blight": {
        "icon": "🟠",
        "description": "Early blight is a fungal disease commonly associated with dark spots and lesions on tomato leaves.",
        "recommendation": "Remove severely affected leaves, maintain good airflow, avoid prolonged leaf wetness, and follow appropriate fungal disease management practices."
    },
    "Tomato_Late_blight": {
        "icon": "🔴",
        "description": "Late blight is a serious disease that can produce dark, water-soaked lesions on tomato leaves.",
        "recommendation": "Remove affected plant material, improve air circulation, avoid overhead watering, and follow recommended disease-management practices."
    },
    "Tomato_healthy": {
        "icon": "🟢",
        "description": "The image appears to show a healthy tomato leaf without the target disease patterns.",
        "recommendation": "Continue good crop management, maintain proper irrigation and nutrition, and regularly inspect plants for symptoms."
    }
}

# ============================================================
# MODEL
# ============================================================

@st.cache_resource
def load_model():
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights=None
    )
    base_model.trainable = False

    data_augmentation = keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
    ], name="data_augmentation")

    inputs = keras.Input(shape=(224, 224, 3), name="image_input")
    x = data_augmentation(inputs)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D(name="global_average_pooling2d")(x)
    x = layers.Dropout(0.2, name="dropout")(x)
    outputs = layers.Dense(3, activation="softmax", name="classifier")(x)

    model = keras.Model(inputs, outputs)

    # The current local project uses this weights file.
    weights_path = os.path.join("model", "plant_disease.weights.h5")
    full_model_path = os.path.join("model", "plant_disease_mobilenetv2.keras")

    if os.path.exists(full_model_path):
        try:
            return keras.models.load_model(full_model_path)
        except Exception:
            pass

    if not os.path.exists(weights_path):
        raise FileNotFoundError(
            "Model weights not found. Put plant_disease.weights.h5 inside the model folder."
        )

    model.load_weights(weights_path)
    return model

try:
    model = load_model()
    MODEL_READY = True
except Exception as e:
    MODEL_READY = False
    MODEL_ERROR = str(e)
    model = None

# ============================================================
# GRAD-CAM
# ============================================================

def make_gradcam(model, image_array, pred_index=None):
    """
    Grad-CAM for the PlantVision AI MobileNetV2 model.

    The important detail is that the target convolution activations and the
    selected prediction are produced during the SAME forward pass. This keeps
    both tensors connected to the same TensorFlow computation graph and avoids
    the "Could not calculate gradients" error.
    """
    try:
        # Locate the nested MobileNetV2 backbone.
        base_model = None
        base_index = None

        for i, layer in enumerate(model.layers):
            if isinstance(layer, keras.Model) and "mobilenetv2" in layer.name.lower():
                base_model = layer
                base_index = i
                break

        if base_model is None:
            try:
                base_model = model.get_layer("mobilenetv2_1.00_224")
                base_index = model.layers.index(base_model)
            except Exception:
                raise ValueError("MobileNetV2 feature extractor could not be located.")

        # Find the last convolutional layer inside MobileNetV2.
        target_layer = None
        target_position = None
        for i in range(len(base_model.layers) - 1, -1, -1):
            layer = base_model.layers[i]
            if isinstance(layer, (layers.Conv2D, layers.DepthwiseConv2D)):
                target_layer = layer
                target_position = i
                break

        if target_layer is None:
            raise ValueError("No convolutional layer was found for Grad-CAM.")

        # ------------------------------------------------------------
        # ONE connected forward pass through the backbone.
        # We do not create a second Model from target_layer.output because
        # that can create a disconnected graph with nested Keras models.
        # ------------------------------------------------------------
        with tf.GradientTape() as tape:
            x = tf.convert_to_tensor(image_array, dtype=tf.float32)
            x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
            conv_outputs = None

            for i, layer in enumerate(base_model.layers):
                # InputLayer does not need to be called manually.
                if isinstance(layer, layers.InputLayer):
                    continue

                x = layer(x, training=False)

                if i == target_position:
                    conv_outputs = x

            if conv_outputs is None:
                raise ValueError("Grad-CAM target activations were not produced.")

            # Continue through the classifier head after MobileNetV2.
            predictions = x
            for layer in model.layers[base_index + 1:]:
                predictions = layer(predictions, training=False)

            if pred_index is None:
                pred_index = tf.argmax(predictions[0])

            pred_index = tf.cast(pred_index, tf.int32)
            class_channel = predictions[:, pred_index]

        grads = tape.gradient(class_channel, conv_outputs)

        if grads is None:
            raise ValueError("Could not calculate gradients for the selected prediction.")

        # Global-average-pool the gradients and weight the feature maps.
        pooled_grads = tf.reduce_mean(grads, axis=(1, 2), keepdims=True)
        heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)
        heatmap = tf.maximum(heatmap, 0)

        max_value = tf.reduce_max(heatmap)
        if float(max_value.numpy()) > 1e-8:
            heatmap = heatmap / max_value
        else:
            raise ValueError("The selected prediction produced an empty Grad-CAM activation map.")

        return heatmap[0].numpy()

    except Exception as e:
        raise ValueError(f"Grad-CAM processing failed: {e}") from e

def overlay_gradcam(original_image, heatmap, alpha=0.42):
    original = np.array(original_image.convert("RGB").resize((224, 224)))
    heatmap_uint8 = np.uint8(255 * heatmap)

    cmap = plt.get_cmap("jet")
    colored = cmap(heatmap_uint8)[:, :, :3]
    colored = np.uint8(colored * 255)

    overlay = np.uint8(
        np.clip(
            original * (1 - alpha) + colored * alpha,
            0,
            255
        )
    )
    return overlay

# ============================================================
# OPTIONAL LOCAL EVALUATION
# ============================================================

@st.cache_data
def evaluate_local_dataset(dataset_path):
    """
    If a local Dataset folder exists, calculate real precision,
    recall, F1 and confusion matrix.

    Expected:
    Dataset/
      Tomato_Early_blight/
      Tomato_Late_blight/
      Tomato_healthy/
    """
    if not os.path.isdir(dataset_path):
        return None

    try:
        ds = tf.keras.utils.image_dataset_from_directory(
            dataset_path,
            image_size=(224, 224),
            batch_size=32,
            shuffle=False
        )

        y_true = []
        y_pred = []

        for images, labels in ds:
            probs = model.predict(images, verbose=0)
            preds = np.argmax(probs, axis=1)
            y_true.extend(labels.numpy())
            y_pred.extend(preds)

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
        recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
        f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        cm = confusion_matrix(y_true, y_pred, labels=np.arange(3))

        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "cm": cm,
            "report": classification_report(
                y_true,
                y_pred,
                target_names=[display_names[x] for x in class_names],
                zero_division=0
            )
        }
    except Exception:
        return None

# ============================================================
# SIDEBAR NAVIGATION
# ============================================================

if "page" not in st.session_state:
    st.session_state.page = "Dashboard"

with st.sidebar:
    st.markdown("""
    <div class="brand">
        <div class="logo-wrap">
            <div class="logo-mark">
                <svg viewBox="0 0 64 64" width="34" height="34" aria-hidden="true">
                    <path d="M53 9C35 10 19 17 13 30c-5 11 2 22 13 22 14 0 23-13 27-43Z"
                          fill="none" stroke="white" stroke-width="4" stroke-linecap="round"/>
                    <path d="M12 53C22 39 31 30 45 21" fill="none"
                          stroke="white" stroke-width="4" stroke-linecap="round"/>
                    <circle cx="19" cy="43" r="4" fill="white"/>
                    <circle cx="31" cy="31" r="4" fill="white"/>
                    <circle cx="44" cy="21" r="4" fill="white"/>
                </svg>
            </div>
            <div>
                <h1>PlantVision AI</h1>
                <p>Computer Vision Intelligence</p>
            </div>
        </div>
    </div>
    <div class="sidebar-label">Workspace</div>
    """, unsafe_allow_html=True)

    if st.button("⌁   Disease Detection", use_container_width=True, key="sidebar_detection"):
        st.session_state.page = "Disease Detection"

    if st.button("⌂   Dashboard", use_container_width=True, key="sidebar_dashboard"):
        st.session_state.page = "Dashboard"

    if st.button("◈   Model Performance", use_container_width=True, key="sidebar_performance"):
        st.session_state.page = "Model Performance"

    st.markdown("""
    <div class="sidebar-panel">
        <div class="row"><span>ENGINE</span><span class="value">MobileNetV2</span></div>
        <div class="row"><span>INPUT</span><span class="value">224 × 224</span></div>
        <div class="row"><span>CLASSES</span><span class="value">03</span></div>
        <div class="row"><span>MODE</span><span class="value">Inference</span></div>
        <div class="row"><span>STATUS</span><span class="value"><span class="live-dot"></span>ONLINE</span></div>
    </div>
    <div class="nav-note">
        ● AI inference engine ready<br>
        Secure local model • v1.0
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# HERO
# ============================================================

st.markdown("""
<div class="hero">
    <h1>🌿 PlantVision AI</h1>
    <p>Deep-learning-based tomato leaf disease detection with explainable predictions.</p>
    <span class="tag">MobileNetV2</span>
    <span class="tag">Image Classification</span>
    <span class="tag">Transfer Learning</span>
    <span class="tag">Grad-CAM</span>
</div>
""", unsafe_allow_html=True)

# ============================================================
# DASHBOARD
# ============================================================

if st.session_state.page == "Dashboard":

    st.markdown('<div class="section-title">🏠 Dashboard</div>', unsafe_allow_html=True)
    st.markdown(
        '<p class="muted">A quick overview of the PlantVision AI project and trained model.</p>',
        unsafe_allow_html=True
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown("""
        <div class="metric">
            <div class="value">95.11%</div>
            <div class="label">Best validation accuracy</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="metric">
            <div class="value">4,500</div>
            <div class="label">Images in dataset</div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown("""
        <div class="metric">
            <div class="value">3</div>
            <div class="label">Disease classes</div>
        </div>
        """, unsafe_allow_html=True)

    with c4:
        st.markdown("""
        <div class="metric">
            <div class="value">224²</div>
            <div class="label">Input resolution</div>
        </div>
        """, unsafe_allow_html=True)

    st.write("")

    left, right = st.columns([1.35, 1])

    with left:
        st.markdown("""
        <div class="card">
            <h3>🌱 Project Overview</h3>
            <p>
            PlantVision AI classifies tomato leaf images into three categories:
            Early Blight, Late Blight, and Healthy. The model uses MobileNetV2
            with ImageNet-pretrained weights as a frozen feature extractor and
            a custom three-class classification head.
            </p>
            <p>
            The training pipeline used image resizing, data augmentation,
            MobileNetV2 preprocessing, transfer learning and softmax
            classification.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with right:
        status = "ONLINE" if MODEL_READY else "OFFLINE"
        st.markdown(f"""
        <div class="card">
            <h3>⚡ Model Status</h3>
            <h2>{status}</h2>
            <p class="muted">MobileNetV2 inference engine</p>
            <p><b>Best validation accuracy:</b> {BEST_VAL_ACC*100:.2f}%</p>
            <p><b>Best epoch:</b> {BEST_EPOCH}</p>
            <p><b>Training images:</b> 3,600</p>
            <p><b>Validation images:</b> 900</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### 🦠 Supported Diseases")

    d1, d2, d3 = st.columns(3)

    with d1:
        st.markdown("""
        <div class="card disease">
            <h3>🟠 Early Blight</h3>
            <p>Fungal disease associated with dark leaf spots and lesions.</p>
        </div>
        """, unsafe_allow_html=True)

    with d2:
        st.markdown("""
        <div class="card disease">
            <h3>🔴 Late Blight</h3>
            <p>Serious disease that can produce dark, water-soaked lesions.</p>
        </div>
        """, unsafe_allow_html=True)

    with d3:
        st.markdown("""
        <div class="card disease">
            <h3>🟢 Healthy</h3>
            <p>Leaf appears free from the target disease patterns.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### 🔄 AI Workflow")

    st.markdown("""
    <div class="card">
        <b>Image</b> → <b>Resize 224×224</b> → <b>Augmentation</b> →
        <b>MobileNetV2</b> → <b>Feature Extraction</b> →
        <b>Global Average Pooling</b> → <b>Dropout</b> →
        <b>Softmax</b> → <b>Disease Prediction</b>
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# DISEASE DETECTION
# ============================================================

elif st.session_state.page == "Disease Detection":

    st.markdown('<div class="section-title">🔬 Disease Detection</div>', unsafe_allow_html=True)
    st.markdown(
        '<p class="muted">Upload a tomato leaf or capture one using your camera.</p>',
        unsafe_allow_html=True
    )

    if not MODEL_READY:
        st.error(f"Model could not be loaded: {MODEL_ERROR}")
        st.stop()

    st.markdown("""
    <div class="info" style="margin: 14px 0 18px 0;">
        <h3 style="margin:0 0 8px 0;">🌿 How to use PlantVision AI</h3>
        <p style="margin:0 0 8px 0;">
            1. Upload a clear tomato leaf image or use your camera.
            &nbsp;&nbsp;→&nbsp;&nbsp;
            2. Click <b>Analyze Leaf</b>.
            &nbsp;&nbsp;→&nbsp;&nbsp;
            3. The model predicts <b>Early Blight, Late Blight, or Healthy</b>.
        </p>
        <p class="small" style="margin:0;">
            The image is resized to <b>224 × 224</b> and processed by the trained
            MobileNetV2 model. The result includes the predicted class,
            confidence, class probabilities, and Grad-CAM explanation.
        </p>
    </div>
    """, unsafe_allow_html=True)

    input_mode = st.radio(
        "Choose image source",
        ["📤 Upload image", "📷 Camera"],
        horizontal=True
    )

    uploaded_file = None
    camera_image = None

    if input_mode == "📤 Upload image":
        uploaded_file = st.file_uploader(
            "Upload a tomato leaf image",
            type=["jpg", "jpeg", "png"],
            help="Use a clear image where the tomato leaf is visible."
        )
        image_source = uploaded_file
    else:
        camera_image = st.camera_input("Take a picture of the tomato leaf")
        image_source = camera_image

    if image_source is not None:

        image = Image.open(image_source).convert("RGB")

        left, right = st.columns([1, 1])

        with left:
            st.markdown("### 📷 Input Leaf")
            st.image(image, use_container_width=True)

        with right:
            st.markdown("### ⚙️ Analysis")
            st.markdown("""
            <div class="info">
                The image will be resized to <b>224 × 224</b> pixels and
                passed through the trained MobileNetV2 classification model.
            </div>
            """, unsafe_allow_html=True)

            analyze = st.button(
                "🔍 Analyze Leaf",
                type="primary",
                use_container_width=True,
                key="analyze_leaf"
            )

        if analyze:

            img_resized = image.resize((224, 224))
            img_array = np.array(img_resized).astype("float32")
            img_batch = np.expand_dims(img_array, axis=0)

            predictions = model.predict(img_batch, verbose=0)[0]

            predicted_index = int(np.argmax(predictions))
            predicted_class = class_names[predicted_index]
            confidence = float(predictions[predicted_index])

            info = disease_info[predicted_class]

            st.markdown("---")
            st.markdown("### 🧠 Prediction Result")

            r1, r2, r3 = st.columns(3)

            with r1:
                st.markdown(f"""
                <div class="metric">
                    <div class="value">{info["icon"]}</div>
                    <div class="label">Detected condition</div>
                    <h3>{display_names[predicted_class]}</h3>
                </div>
                """, unsafe_allow_html=True)

            with r2:
                st.markdown(f"""
                <div class="metric">
                    <div class="value">{confidence*100:.2f}%</div>
                    <div class="label">Model confidence</div>
                    <h3>Prediction confidence</h3>
                </div>
                """, unsafe_allow_html=True)

            with r3:
                st.markdown("""
                <div class="metric">
                    <div class="value">3</div>
                    <div class="label">Possible classes</div>
                    <h3>Classification</h3>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("")

            st.markdown(f"""
            <div class="result">
                <h2>{info["icon"]} {display_names[predicted_class]}</h2>
                <p>{info["description"]}</p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("### 🌱 Farmer Recommendation")
            st.markdown(f"""
            <div class="card">
                <p>{info["recommendation"]}</p>
                <p class="small">
                Note: This application is an AI screening/educational tool and
                should not replace professional agricultural diagnosis.
                </p>
            </div>
            """, unsafe_allow_html=True)

            # Probability chart
            st.markdown("### 📊 Class Probabilities")

            fig, ax = plt.subplots(figsize=(9, 3.8))
            labels = [display_names[x] for x in class_names]
            ax.bar(labels, predictions * 100)
            ax.set_ylabel("Probability (%)")
            ax.set_ylim(0, 100)
            ax.set_title("Model prediction distribution")
            ax.tick_params(axis="x", rotation=12)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

            # Grad-CAM
            st.markdown("### 🧠 Grad-CAM Explainability")
            st.markdown(
                "Grad-CAM highlights image regions that contributed to the model's selected prediction.",
                unsafe_allow_html=True
            )

            try:
                heatmap = make_gradcam(
                    model,
                    img_batch,
                    pred_index=predicted_index
                )
                overlay = overlay_gradcam(image, heatmap)

                g1, g2 = st.columns(2)

                with g1:
                    st.image(
                        image.resize((224, 224)),
                        caption="Original image",
                        use_container_width=True
                    )

                with g2:
                    st.image(
                        overlay,
                        caption="Grad-CAM attention map",
                        use_container_width=True
                    )

            except Exception as e:
                st.warning(
                    "Grad-CAM could not be generated for this model file. "
                    f"Technical detail: {e}"
                )

            # ------------------------------------------------------------
            # CSV PREDICTION REPORT
            # ------------------------------------------------------------
            runner_up_indices = np.argsort(predictions)[::-1]
            runner_up_index = int(runner_up_indices[1])
            runner_up_class = class_names[runner_up_index]
            runner_up_confidence = float(predictions[runner_up_index])
            confidence_gap = confidence - runner_up_confidence

            if confidence >= 0.80:
                confidence_status = "High confidence"
            elif confidence >= 0.60:
                confidence_status = "Moderate confidence"
            else:
                confidence_status = "Low confidence - review image"

            st.markdown("### 📥 Download Prediction Report")
            prediction_report = pd.DataFrame([
                {
                    "Prediction Date": pd.Timestamp.now().strftime("%Y-%m-%d"),
                    "Prediction Time": pd.Timestamp.now().strftime("%H:%M:%S"),
                    "Predicted Disease": display_names[predicted_class],
                    "Model Confidence (%)": round(confidence * 100, 2),
                    "Early Blight Probability (%)": round(float(predictions[0]) * 100, 2),
                    "Late Blight Probability (%)": round(float(predictions[1]) * 100, 2),
                    "Healthy Probability (%)": round(float(predictions[2]) * 100, 2),
                    "Runner-up Prediction": display_names[runner_up_class],
                    "Runner-up Confidence (%)": round(runner_up_confidence * 100, 2),
                    "Confidence Gap (%)": round(confidence_gap * 100, 2),
                    "Confidence Status": confidence_status
                }
            ])

            csv_report = prediction_report.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="⬇️ Download Prediction Report (CSV)",
                data=csv_report,
                file_name="plantvision_prediction_report.csv",
                mime="text/csv",
                use_container_width=True,
                key="download_prediction_report"
            )

# ============================================================
# MODEL PERFORMANCE
# ============================================================

elif st.session_state.page == "Model Performance":

    st.markdown('<div class="section-title">📊 Model Performance</div>', unsafe_allow_html=True)
    st.markdown(
        '<p class="muted">Training results and evaluation metrics from the PlantVision AI training pipeline.</p>',
        unsafe_allow_html=True
    )

    # Actual metrics available from notebook
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown(f"""
        <div class="metric">
            <div class="value">{BEST_VAL_ACC*100:.2f}%</div>
            <div class="label">Best validation accuracy</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="metric">
            <div class="value">{TRAIN_ACC[-1]*100:.2f}%</div>
            <div class="label">Last recorded training accuracy</div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div class="metric">
            <div class="value">{VAL_LOSS[-1]:.4f}</div>
            <div class="label">Last recorded validation loss</div>
        </div>
        """, unsafe_allow_html=True)

    with c4:
        st.markdown(f"""
        <div class="metric">
            <div class="value">{BEST_EPOCH}</div>
            <div class="label">Best validation epoch</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### 📈 Training Accuracy")

    fig1, ax1 = plt.subplots(figsize=(9, 4))
    ax1.plot(EPOCHS, np.array(TRAIN_ACC) * 100, marker="o", label="Training Accuracy")
    ax1.plot(EPOCHS, np.array(VAL_ACC) * 100, marker="o", label="Validation Accuracy")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy (%)")
    ax1.set_title("Training vs Validation Accuracy")
    ax1.set_xticks(EPOCHS)
    ax1.legend()
    ax1.grid(alpha=.2)
    st.pyplot(fig1, use_container_width=True)
    plt.close(fig1)

    st.markdown("### 📉 Training Loss")

    fig2, ax2 = plt.subplots(figsize=(9, 4))
    ax2.plot(EPOCHS, TRAIN_LOSS, marker="o", label="Training Loss")
    ax2.plot(EPOCHS, VAL_LOSS, marker="o", label="Validation Loss")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.set_title("Training vs Validation Loss")
    ax2.set_xticks(EPOCHS)
    ax2.legend()
    ax2.grid(alpha=.2)
    st.pyplot(fig2, use_container_width=True)
    plt.close(fig2)

    st.markdown("### 🎯 Precision, Recall & F1 Score")

    # Search for a local Dataset folder. If present, calculate real metrics.
    possible_paths = [
        "Dataset",
        "dataset",
        os.path.join("data", "Dataset"),
        os.path.join("data", "dataset")
    ]

    evaluation = None
    for path in possible_paths:
        if os.path.isdir(path):
            evaluation = evaluate_local_dataset(path)
            if evaluation is not None:
                break

    if evaluation is not None:

        p1, p2, p3 = st.columns(3)

        with p1:
            st.metric("Macro Precision", f"{evaluation['precision']*100:.2f}%")

        with p2:
            st.metric("Macro Recall", f"{evaluation['recall']*100:.2f}%")

        with p3:
            st.metric("Macro F1 Score", f"{evaluation['f1']*100:.2f}%")

        st.markdown("### 🔲 Confusion Matrix")

        fig3, ax3 = plt.subplots(figsize=(7, 5))
        im = ax3.imshow(evaluation["cm"], interpolation="nearest")
        ax3.set_title("Confusion Matrix")
        ax3.set_xlabel("Predicted label")
        ax3.set_ylabel("True label")
        ax3.set_xticks(range(3))
        ax3.set_yticks(range(3))
        ax3.set_xticklabels([display_names[x] for x in class_names], rotation=25, ha="right")
        ax3.set_yticklabels([display_names[x] for x in class_names])

        for i in range(3):
            for j in range(3):
                ax3.text(j, i, evaluation["cm"][i, j], ha="center", va="center")

        fig3.colorbar(im, ax=ax3)
        st.pyplot(fig3, use_container_width=True)
        plt.close(fig3)

        with st.expander("View classification report"):
            st.code(evaluation["report"])

    else:
        st.info(
            "Precision, Recall, F1 and Confusion Matrix are not present in the "
            "uploaded training notebook. The notebook contains training/validation "
            "accuracy and loss, but no test-set classification report. "
            "If you place the original Dataset folder beside app.py, this page "
            "will calculate these metrics automatically."
        )

    st.markdown("### 🧠 Model Architecture")

    st.markdown("""
    <div class="card">
        <p><b>Input:</b> 224 × 224 × 3 RGB image</p>
        <p>↓</p>
        <p><b>Data Augmentation:</b> Random Flip + Random Rotation + Random Zoom</p>
        <p>↓</p>
        <p><b>MobileNetV2:</b> ImageNet-pretrained feature extractor, frozen during training</p>
        <p>↓</p>
        <p><b>Global Average Pooling:</b> 1280 features</p>
        <p>↓</p>
        <p><b>Dropout:</b> 0.20</p>
        <p>↓</p>
        <p><b>Dense:</b> 3 output neurons</p>
        <p>↓</p>
        <p><b>Softmax:</b> Three-class probability distribution</p>
        <hr>
        <p><b>Total parameters:</b> 2,261,827</p>
        <p><b>Trainable parameters:</b> 3,843</p>
        <p><b>Non-trainable parameters:</b> 2,257,984</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### ⚙️ Training Configuration")

    t1, t2, t3, t4 = st.columns(4)

    with t1:
        st.metric("Optimizer", "Adam")

    with t2:
        st.metric("Learning rate", "0.001")

    with t3:
        st.metric("Loss", "Sparse CCE")

    with t4:
        st.metric("Batch size", "32")

    st.markdown("""
    <div class="info">
        <b>Training setup:</b> 4,500 images were split into 3,600 training
        images and 900 validation images using an 80/20 split with seed 42.
        Training was configured for up to 8 epochs with EarlyStopping
        (patience=2) and a best-model checkpoint monitored on validation accuracy.
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# FOOTER
# ============================================================

st.markdown("""
<br>
<hr>
<p style="text-align:center;color:#738078;font-size:12px;">
PlantVision AI • Deep Learning • TensorFlow/Keras • MobileNetV2 • Streamlit
</p>
""", unsafe_allow_html=True)
