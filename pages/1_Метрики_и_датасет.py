import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

st.set_page_config(page_title="Метрики модели", page_icon="📊", layout="wide")
st.title("📊 Обучение и качество модели")

path = Path("artifacts/metrics.json")
if not path.exists():
    st.warning("Файл artifacts/metrics.json не найден. Сначала обучите модель командой `python train.py --data_dir data`.")
    st.stop()

m = json.loads(path.read_text(encoding="utf-8"))
st.subheader("Состав датасета")
c1, c2, c3 = st.columns(3)
c1.metric("Обучающих изображений", f"{m['train_size']:,}".replace(",", " "))
c2.metric("Тестовых изображений", f"{m['test_size']:,}".replace(",", " "))
c3.metric("Количество классов", len(m["classes"]))

counts = pd.DataFrame({
    "Класс": m["classes"],
    "Train": [m["class_counts_train"][c] for c in m["classes"]],
    "Test": [m["class_counts_test"][c] for c in m["classes"]],
}).set_index("Класс")
st.bar_chart(counts)
st.dataframe(counts, use_container_width=True)

st.subheader("Кривые обучения")
hist = m["history"]
epochs = list(range(1, len(hist["train_losses"]) + 1))
left, right = st.columns(2)
with left:
    fig, ax = plt.subplots()
    ax.plot(epochs, hist["train_losses"], marker="o", label="Train")
    ax.plot(epochs, hist["valid_losses"], marker="o", label="Test")
    ax.set(xlabel="Эпоха", ylabel="Loss", title="Функция потерь")
    ax.grid(alpha=.3); ax.legend()
    st.pyplot(fig); plt.close(fig)
with right:
    fig, ax = plt.subplots()
    ax.plot(epochs, hist["train_accs"], marker="o", label="Train")
    ax.plot(epochs, hist["valid_accs"], marker="o", label="Test")
    ax.set(xlabel="Эпоха", ylabel="Accuracy", title="Точность")
    ax.set_ylim(0, 1); ax.grid(alpha=.3); ax.legend()
    st.pyplot(fig); plt.close(fig)

st.subheader("Итоговые метрики")
a, b, c = st.columns(3)
a.metric("Macro F1-score", f"{m['f1_macro']:.4f}")
b.metric("Время обучения", f"{m['training_time_seconds']:.1f} сек.")
c.metric("Эпох", m["epochs"])

st.subheader("Confusion matrix")
cm = pd.DataFrame(m["confusion_matrix"], index=m["classes"], columns=m["classes"])
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=m["classes"],
            yticklabels=m["classes"], ax=ax)
ax.set_xlabel("Предсказанный класс")
ax.set_ylabel("Истинный класс")
st.pyplot(fig); plt.close(fig)

with st.expander("Подробный classification report"):
    report = pd.DataFrame(m["classification_report"]).T
    st.dataframe(report, use_container_width=True)
