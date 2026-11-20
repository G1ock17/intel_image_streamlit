import io
import os
import time
import requests
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms as T
from torchvision.models import resnet50

st.set_page_config(page_title="Intel Image Classification", page_icon="🖼️", layout="wide")

CLASS_NAMES = ["buildings", "forest", "glacier", "mountain", "sea", "street"]
MODEL_PATH = "artifacts/resnet50_intel.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def load_model():
    model = resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
    model.load_state_dict(checkpoint)
    model.to(DEVICE).eval()
    return model

preprocess = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])

def classify(image, model):
    image = image.convert("RGB")
    tensor = preprocess(image).unsqueeze(0).to(DEVICE)
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.inference_mode():
        probabilities = torch.softmax(model(tensor), dim=1)[0]
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    confidence, index = probabilities.max(dim=0)
    return CLASS_NAMES[index.item()], confidence.item(), probabilities.cpu().tolist(), elapsed

st.title("🖼️ Intel Image Classification")
st.write("Классификация изображений с помощью ResNet50 (Transfer Learning).")
st.caption(f"Устройство: {DEVICE}")

if not os.path.exists(MODEL_PATH):
    st.error("Не найдены обученные веса модели: artifacts/resnet50_intel.pth")
    st.info("Сначала запустите обучение: `python train.py --data_dir data`.")
    st.stop()

model = load_model()
tab1, tab2 = st.tabs(["Загрузка изображений", "По ссылке"])

with tab1:
    uploaded = st.file_uploader(
        "Выберите одно или несколько изображений",
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=True
    )
    if uploaded:
        for file in uploaded:
            try:
                image = Image.open(file).convert("RGB")
                with st.spinner(f"Классификация {file.name}..."):
                    label, confidence, probs, elapsed = classify(image, model)
                left, right = st.columns([1, 1])
                with left:
                    st.image(image, caption=file.name, use_container_width=True)
                with right:
                    st.subheader(f"Класс: {label}")
                    st.metric("Уверенность", f"{confidence:.2%}")
                    st.metric("Время ответа модели", f"{elapsed:.4f} сек.")
                    st.bar_chart({name: prob for name, prob in zip(CLASS_NAMES, probs)})
                st.divider()
            except Exception as exc:
                st.warning(f"Не удалось обработать {file.name}: {exc}")

with tab2:
    url = st.text_input("Вставьте прямую ссылку на изображение", placeholder="https://example.com/image.jpg")
    if st.button("Загрузить и классифицировать", type="primary"):
        if not url.strip():
            st.warning("Введите ссылку на изображение.")
        else:
            try:
                response = requests.get(url.strip(), timeout=20, headers={"User-Agent": "Mozilla/5.0"})
                response.raise_for_status()
                image = Image.open(io.BytesIO(response.content)).convert("RGB")
                label, confidence, probs, elapsed = classify(image, model)
                left, right = st.columns([1, 1])
                with left:
                    st.image(image, caption="Изображение по ссылке", use_container_width=True)
                with right:
                    st.subheader(f"Класс: {label}")
                    st.metric("Уверенность", f"{confidence:.2%}")
                    st.metric("Время ответа модели", f"{elapsed:.4f} сек.")
                    st.bar_chart({name: prob for name, prob in zip(CLASS_NAMES, probs)})
            except Exception as exc:
                st.error(f"Не удалось загрузить или классифицировать изображение: {exc}")
