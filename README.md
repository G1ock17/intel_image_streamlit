# Intel Image Classification — Streamlit + ResNet50

## Структура датасета

Положите Intel Image Classification в `data/`. Поддерживаются оба распространённых варианта:

```text
data/
  seg_train/
    buildings/
    forest/
    glacier/
    mountain/
    sea/
    street/
  seg_test/
    buildings/
    ...
```

или вложенный вариант `data/seg_train/seg_train/<class>` и `data/seg_test/seg_test/<class>`.

## Локальный запуск

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt

python train.py --data_dir data --epochs 5 --batch_size 32 --lr 0.001
streamlit run app.py
```

Обучение скачивает ImageNet-веса ResNet50 при первом запуске. После обучения создаются:
- `artifacts/resnet50_intel.pth` — веса модели;
- `artifacts/metrics.json` — история обучения, размеры выборок, распределение классов, Macro F1 и confusion matrix.

## Публикация на Streamlit Community Cloud

1. Создайте GitHub-репозиторий и загрузите файлы проекта.
2. Один раз выполните обучение локально на полном датасете.
3. Добавьте `artifacts/resnet50_intel.pth` и `artifacts/metrics.json` в репозиторий. Если весовой файл превышает лимит GitHub, используйте Git LFS или внешнее хранилище и скачивание артефакта при старте приложения.
4. Откройте https://share.streamlit.io/, подключите GitHub и выберите `app.py` как Main file path.
5. В Advanced settings выберите подходящую версию Python и разверните приложение.

Датасет обычно не нужно публиковать вместе с приложением: он требуется для обучения, но не для инференса. Не коммитьте `data/`.

## Примечания

- Для предсказания используется ImageNet-нормализация, соответствующая предобученной ResNet.
- Все слои backbone заморожены, обучается только `fc`.
- Время инференса измеряется отдельно для каждого изображения, без времени загрузки файла.
- Если обучаете на CPU, уменьшите batch size; на GPU обучение будет заметно быстрее.
