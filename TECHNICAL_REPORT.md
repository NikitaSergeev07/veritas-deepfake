# Veritas — Технический отчёт

## 1. Обзор системы

Veritas — мультимодальный веб-сервис для обнаружения дипфейков в трёх форматах контента: изображения, аудио, видео. Система состоит из Vue 3 фронтенда, FastAPI бэкенда и трёх независимых ML-детекторов.

Весь анализ выполняется **локально** на устройстве пользователя. Файлы не загружаются на внешние серверы — бэкенд запускается на localhost, ML-модели скачиваются один раз из HuggingFace Hub и кешируются на диск. Это гарантирует приватность данных.

---

## 2. Image Detector

### 2.1 Архитектура

- **Backbone**: CLIP ViT-B/16 (OpenAI) — vision transformer, 150M параметров, замороженный
- **Head**: MLP 512 → 256 → 1 (132K обучаемых параметров)
  - LayerNorm(512) → Linear(512, 256) → GELU → Dropout(0.3) → Linear(256, 1)
- **Общий подход**: UniversalFakeDetect (CVPR 2023) — замороженные CLIP-фичи + лёгкий классификатор

### 2.2 Предобученная модель

CLIP ViT-B/16 загружается из библиотеки `open_clip` с весами `openai`. Visual encoder замораживается полностью (`requires_grad_(False)`), используется только как feature extractor. Обучается только MLP-голова.

### 2.3 Обучение MLP Head

**Двухэтапный процесс:**

1. **Извлечение фичей** (`extract_features.py`):
   - Все изображения из датасета пропускаются через замороженный CLIP ViT-B/16
   - Каждое изображение даёт 512-мерный вектор фичей
   - Для train set: 2 аугментированных копии каждого изображения (RandomResizedCrop, HorizontalFlip, ColorJitter, JPEG Compression, Gaussian Noise)
   - Для val set: чистые трансформации (Resize + CenterCrop)
   - Фичи сохраняются на диск как .pt файлы

2. **Обучение MLP** (`train.py`):
   - Загрузка кешированных фичей (без повторного запуска CLIP)
   - Optimizer: AdamW (lr=1e-3, weight_decay=0.01)
   - Scheduler: CosineAnnealingLR
   - Loss: BCEWithLogitsLoss с pos_weight (балансировка классов)
   - Early stopping: patience=10 по val AUC-ROC
   - До 50 эпох
   - Калибровка порога: перебор thresholds 0.01–0.99, максимизация balanced accuracy

### 2.4 Датасеты для обучения

Три источника из HuggingFace:
- `Hemg/deepfake-and-real-images` (~190K пар, label: 0=Fake, 1=Real)
- `itsLeen/deepfake_vs_real_image_detection` (~7.4K, ClassLabel)
- `date3k2/raw_real_fake_images` (~9.3K, ClassLabel)

Автоматическое определение меток через ClassLabel.names или заданные маппинги.

### 2.5 Multi-view инференс

При инференсе каждое изображение анализируется тремя способами:
1. **full_frame** — полное изображение (Resize 224 + CenterCrop 224)
2. **center_square** — центральный квадратный кроп
3. **focus_crop** — центральные 82% изображения

Для каждого view CLIP извлекает фичи → MLP выдаёт P(fake). Финальный вердикт — среднее трёх вероятностей. Это повышает устойчивость: артефакты могут быть как в центре (face swap), так и по краям (генеративные модели).

Дополнительно: MediaPipe face detection для детекции лица — если лицо найдено, анализируется кроп лица.

### 2.6 Метрики

- **Accuracy**: ~94% на валидационной выборке
- **AUC-ROC**: сохраняется в чекпоинте
- **Оптимальный порог**: калибруется автоматически по balanced accuracy

---

## 3. Audio Detector

### 3.1 Архитектура

Двухуровневый стекинг ансамбль:

**Уровень 1 — 5 базовых моделей** (предобученные Wav2Vec2 из HuggingFace):
1. `garystafford/wav2vec2-deepfake-voice-detector`
2. `MelodyMachine/Deepfake-audio-detection`
3. `Hemgg/Deepfake-audio-detection`
4. `Gustking/wav2vec2-large-xlsr-deepfake-audio-classification`
5. `mo-thecreator/Deepfake-audio-detection`

Каждая модель — это fine-tuned Wav2Vec2 для бинарной классификации audio (real vs fake). Используются как есть через `transformers.pipeline("audio-classification")`.

**Уровень 2 — мета-классификатор:**
- StandardScaler (нормализация 5 скоров)
- LogisticRegression (C=1.0, solver=lbfgs, max_iter=1000, penalty=l2, random_state=42)

### 3.2 Обучение мета-классификатора

Мета-классификатор (LogisticRegression) **обучался нами** на датасете `garystafford/deepfake-audio-detection`:
- Источники фейков: ElevenLabs, Kokoro, Amazon Polly, Hume AI
- Реальные записи: YouTube clips

Процесс:
1. Все 5 базовых моделей прогоняются на каждом аудио из датасета
2. Для каждого аудио получается вектор из 5 fake_probability скоров
3. На этих 5-мерных векторах обучается StandardScaler + LogisticRegression
4. Сохранённые артефакты: `logistic_regression_model.joblib`, `scaler.joblib`, `feature_names.joblib`, `model_metadata.json`

### 3.3 Извлечение fake_probability из базовых моделей

Каждая базовая модель возвращает список `[{label, score}, ...]`. Нормализация:
- По умолчанию: берётся `result[0]["score"]` (top prediction confidence)
- Если среди меток есть "real": инвертируется (`1 - score`)
- Это гарантирует, что выходное значение = P(fake) для всех моделей, независимо от их формата меток

### 3.4 Калиброванные пороги

Для снижения ложных срабатываний (например, на телефонных записях с шумом):
- **FAKE**: fake_probability >= 0.70
- **REAL**: fake_probability <= 0.35
- **UNCERTAIN**: между 0.35 и 0.70

Зона uncertain позволяет честно сообщать пользователю, когда модель не уверена.

### 3.5 Конвертация форматов

Wav2Vec2 pipelines не поддерживают M4A/OGG напрямую. Бэкенд автоматически конвертирует:
- M4A, MP4, OGG → WAV (16kHz, mono) через ffmpeg
- WAV, MP3, FLAC — передаются напрямую

### 3.6 Метрики

На валидационной выборке (n=1867, сбалансированная: 931 real + 936 fake):

| Метрика | Значение |
|---------|----------|
| **Accuracy** | 98.18% |
| **Precision** | 98.39% |
| **Recall** | 97.97% |
| **F1-Score** | 98.18% |

---

## 4. Video Detector

### 4.1 Архитектура

- **Модель**: VideoMAE-Base (`Vansh180/VideoMae-ffc23-deepfake-detector`)
- **Параметры**: 86M
- **Тип**: Video transformer (masked autoencoder, fine-tuned для классификации)
- **Обучение**: FaceForensics++ C23 (face-swap deepfakes) — **предобученная**, не дообучалась нами
- **Метки**: {0: "real", 1: "fake"}

### 4.2 Pipeline инференса

1. **Извлечение кадров**: ffmpeg с частотой 2 fps → список PIL Image
2. **Face crop**: MediaPipe FaceDetection (model_selection=1, confidence=0.4)
   - Детектируется лицо на среднем кадре
   - Относительные координаты лица применяются ко всем кадрам (temporal consistency)
   - Расширение bbox: ×1.4 от размера лица
   - Fallback: center crop если лицо не найдено
3. **Sampling**: из N кадров выбирается 16 равномерно распределённых (np.linspace)
   - Если кадров < 16 — последний дублируется до 16
4. **VideoMAE inference**: 16 кадров → VideoMAEImageProcessor → VideoMAEForVideoClassification → softmax → P(fake)

### 4.3 Сегментный анализ

Для видео длиннее 32 кадров (16 секунд при 2fps):
- Видео разбивается на сегменты по 16 кадров (до 6 сегментов максимум)
- Каждый сегмент анализируется независимо через VideoMAE
- Для каждого сегмента: start_sec, end_sec, fake_probability, label
- В UI: цветовая шкала-таймлайн (зелёный=real, красный=fake, жёлтый=uncertain)

### 4.4 Пороги

- **FAKE**: fake_probability >= 0.65
- **REAL**: fake_probability <= 0.35
- **UNCERTAIN**: между 0.35 и 0.65

Более низкий порог fake (0.65 vs 0.70 у аудио) — VideoMAE обучен на конкретном домене (FF++ face-swap), уверенность модели выше на этом типе фейков.

### 4.5 Метрики

На тестовых клипах из FaceForensics++ и DFDC:
- **Accuracy**: ~93% на FF++ C23 тестовой выборке
- Модель хорошо работает на face-swap дипфейках (DeepFaceLab, FaceSwap)
- Ограничение: на дипфейках другого типа (lip-sync, full body) точность ниже

---

## 5. Backend

### 5.1 Архитектура

- **Framework**: FastAPI с asyncio
- **Параллелизм**: ThreadPoolExecutor для каждого детектора
  - Image: 2 воркера
  - Audio: 2 воркера
  - Video: 1 воркер (тяжёлый инференс)
- **Lazy loading**: модели загружаются при первом запросе, не при старте сервера
- **Thread safety**: threading.Lock для защиты инициализации моделей

### 5.2 API Endpoints

| Endpoint | Метод | Описание |
|----------|-------|----------|
| `/health` | GET | Статус image detector |
| `/health/audio` | GET | Статус audio detector |
| `/health/video` | GET | Статус video detector |
| `/predict` | POST | Анализ изображения |
| `/predict/audio` | POST | Анализ аудио |
| `/predict/video` | POST | Анализ видео |
| `/inspect` | POST | Метаданные изображения без анализа |

### 5.3 Валидация

- Image: JPG, PNG, WEBP, BMP, TIFF; до 10 МБ
- Audio: WAV, MP3, FLAC, OGG, M4A; до 50 МБ
- Video: MP4, MOV, AVI, MKV, WEBM; до 100 МБ

### 5.4 Приватность

Вся обработка происходит на localhost:
- Бэкенд: `uvicorn backend.app:app --host 0.0.0.0 --port 8000`
- Фронтенд: `npm run dev` → localhost:5173
- ML-модели: скачиваются из HuggingFace Hub при первом запуске, кешируются в `~/.cache/huggingface/`
- Файлы пользователя: загружаются в оперативную память, обрабатываются и удаляются. Не сохраняются на диск (кроме временных файлов ffmpeg, которые удаляются в finally-блоке)
- Никаких внешних API вызовов во время анализа

---

## 6. Frontend

### 6.1 Стек

- Vue 3 (Composition API, `<script setup>`)
- TypeScript
- Vite (dev server + build)
- Единый файл `App.vue` (~970 строк) — SPA без роутера

### 6.2 Функционал

- Три вкладки: Image / Audio / Video
- Drag & drop загрузка файлов
- Визуализация результатов: вероятностные шкалы, цветовая индикация (зелёный/красный/жёлтый)
- Per-view breakdown для изображений (full_frame, center_square, focus_crop)
- Per-model breakdown для аудио (5 базовых моделей)
- Segment timeline для видео (цветовая шкала + детали по сегментам)
- SHA-256 хеш загруженного файла
- Двуязычный интерфейс (RU/EN) — встроенный i18n

---

## 7. Зависимости

Основные Python-пакеты (`requirements.txt`):
- `torch`, `torchvision` — PyTorch
- `transformers` — HuggingFace (Wav2Vec2, VideoMAE)
- `open-clip-torch` — CLIP ViT-B/16
- `scikit-learn` — LogisticRegression, StandardScaler, метрики
- `mediapipe` — face detection
- `fastapi`, `uvicorn` — backend
- `Pillow` — работа с изображениями
- `joblib` — сериализация sklearn моделей
- Внешняя утилита: `ffmpeg` (должен быть установлен в системе)

---

## 8. Ограничения

- **Audio**: модели обучены на TTS-дипфейках (ElevenLabs и пр.). На телефонных записях с шумом могут давать ложные срабатывания — для этого введена зона uncertain
- **Video**: VideoMAE обучен на face-swap (FaceForensics++). На lip-sync или full-body deepfakes точность ниже
- **Image**: CLIP-based подход хорошо работает на AI-generated изображениях, но может быть менее точен на старых методах обработки фото (Photoshop)
- **Производительность**: первый запуск загружает модели из HuggingFace (~5-10 минут). Последующие запуски используют кеш
- **GPU**: поддерживается CUDA. На CPU работает, но инференс медленнее (особенно video). Apple MPS поддерживается для image detector
