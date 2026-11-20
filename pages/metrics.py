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
    st.warning(
        "Файл artifacts/metrics.json не найден. Сначала обучите модель командой "
        "`python train.py --data_dir data`."
    )
    st.stop()

m = json.loads(path.read_text(encoding="utf-8"))

# --- Временная подсказка: посмотреть, какие ключи реально есть в JSON ---
# Раскомментируйте, если снова будет KeyError, и сверьте имена ключей:
# with st.expander("🔧 Отладка: ключи в metrics.json"):
#     st.write(list(m.keys()))
#     st.json(m)

st.subheader("Состав датасета")

# Безопасное получение значений: если ключа нет — 0 / пустой список
train_size = m.get("train_size", m.get("train_count", m.get("n_train", 0)))
test_size = m.get("test_size", m.get("valid_size", m.get("val_size", m.get("n_test", 0))))
classes = m.get("classes", [])

c1, c2, c3 = st.columns(3)
c1.metric("Обучающих изображений", f"{train_size:,}".replace(",", " "))
c2.metric("Тестовых изображений", f"{test_size:,}".replace(",", " "))
c3.metric("Количество классов", len(classes))

# --- Распределение классов по сплитам ---
counts_train = m.get("class_counts_train", {})
counts_test = m.get("class_counts_test", m.get("class_counts_valid", {}))

if classes and counts_train and counts_test:
    counts = pd.DataFrame({
        "Класс": classes,
        "Train": [counts_train.get(c, 0) for c in classes],
        "Test": [counts_test.get(c, 0) for c in classes],
    }).set_index("Класс")
    st.bar_chart(counts)
    st.dataframe(counts, use_container_width=True)
else:
    st.info(
        "В metrics.json нет данных о распределении классов по сплитам "
        "(нужны ключи `class_counts_train` и `class_counts_test`)."
    )

# --- Кривые обучения ---
st.subheader("Кривые обучения")
hist = m.get("history", {})
train_losses = hist.get("train_losses", [])
valid_losses = hist.get("valid_losses", [])
train_accs = hist.get("train_accs", [])
valid_accs = hist.get("valid_accs", [])

if train_losses and valid_losses:
    epochs = list(range(1, len(train_losses) + 1))
    left, right = st.columns(2)

    with left:
        fig, ax = plt.subplots()
        ax.plot(epochs, train_losses, marker="o", label="Train")
        ax.plot(epochs, valid_losses, marker="o", label="Test")
        ax.set(xlabel="Эпоха", ylabel="Loss", title="Функция потерь")
        ax.grid(alpha=.3)
        ax.legend()
        st.pyplot(fig)
        plt.close(fig)

    with right:
        fig, ax = plt.subplots()
        ax.plot(epochs, train_accs, marker="o", label="Train")
        ax.plot(epochs, valid_accs, marker="o", label="Test")
        ax.set(xlabel="Эпоха", ylabel="Accuracy", title="Точность")
        ax.set_ylim(0, 1)
        ax.grid(alpha=.3)
        ax.legend()
        st.pyplot(fig)
        plt.close(fig)
else:
    st.info("В metrics.json нет истории обучения (`history.train_losses` и т.д.).")

# --- Итоговые метрики ---
st.subheader("Итоговые метрики")
a, b, c = st.columns(3)

f1_macro = m.get("f1_macro")
a.metric("Macro F1-score", f"{f1_macro:.4f}" if isinstance(f1_macro, (int, float)) else "—")

t_time = m.get("training_time_seconds")
b.metric("Время обучения", f"{t_time:.1f} сек." if isinstance(t_time, (int, float)) else "—")

c.metric("Эпох", m.get("epochs", "—"))

# --- Confusion matrix ---
st.subheader("Confusion matrix")
cm_raw = m.get("confusion_matrix")
if cm_raw and classes:
    cm = pd.DataFrame(cm_raw, index=classes, columns=classes)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=classes, yticklabels=classes, ax=ax,
    )
    ax.set_xlabel("Предсказанный класс")
    ax.set_ylabel("Истинный класс")
    st.pyplot(fig)
    plt.close(fig)
else:
    st.info("В metrics.json нет confusion matrix (`confusion_matrix`) или классов.")

# --- Classification report ---
with st.expander("Подробный classification report"):
    report_raw = m.get("classification_report")
    if report_raw:
        report = pd.DataFrame(report_raw).T
        st.dataframe(report, use_container_width=True)
    else:
        st.info("В metrics.json нет `classification_report`.")