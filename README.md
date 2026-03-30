# Veritas — Deepfake Detection Service

Веб-сервис для обнаружения дипфейков в изображениях, аудио и видео с помощью ансамблевых AI-моделей. Весь анализ выполняется локально на устройстве пользователя.

## Архитектура

```
                      ┌─────────────────────┐
                      │   Vue 3 Frontend    │
                      │   (VirusTotal UI)   │
                      └─────────┬───────────┘
                                │ HTTP
                      ┌─────────▼───────────┐
                      │   FastAPI Backend   │
                      │   (async + pools)   │
                      └─────────┬───────────┘
                ┌───────────────┼───────────────┐
                ▼               ▼               ▼
        ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
        │    Image     │ │    Audio     │ │    Video     │
        │  Detector    │ │  Detector    │ │  Detector    │
        │              │ │              │ │              │
        │ CLIP ViT-B/16│ │ 5× Wav2Vec2 │ │   VideoMAE   │
        │ + MLP Head   │ │ + Stacking   │ │ (FF++ C23)   │
        │              │ │ LogRegress   │ │ + MediaPipe  │
        └──────────────┘ └──────────────┘ └──────────────┘
```

### Модули детекции

| Модуль | Модель | Подход |
|--------|--------|--------|
| **Image** | CLIP ViT-B/16 + MLP (132K params) | 3 вида (full, center, focus) → среднее |
| **Audio** | 5× Wav2Vec2 + LogisticRegression | Стекинг ансамбль, accuracy 98.2% |
| **Video** | VideoMAE (86M params, FaceForensics++) | Temporal transformer, 16 кадров + face crop |

## Стек технологий

**Frontend:** Vue 3, TypeScript, Vite
**Backend:** FastAPI, uvicorn, asyncio + ThreadPoolExecutor
**AI/ML:** PyTorch, HuggingFace Transformers, open-clip-torch, scikit-learn, MediaPipe
**Утилиты:** ffmpeg (конвертация аудио/видео), Pillow

## Быстрый старт

### 1. Backend

```bash
cd deepfake_detector

# Создать виртуальное окружение
python3 -m venv .venv
source .venv/bin/activate

# Установить зависимости
pip install -r requirements.txt

# Запустить сервер
uvicorn backend.app:app --reload --host 0.0.0.0 --port 8000
```

### 2. Frontend

```bash
cd frontend

# Установить зависимости
npm install

# Запустить dev-сервер
npm run dev
```

Открыть http://localhost:5173 в браузере.

### 3. Скачать тестовые данные (опционально)

```bash
# Аудио (garystafford/deepfake-audio-detection)
python -m ai.audio_detector.download_data

# Видео
python -m ai.video_detector.download_data
```

## Структура проекта

```
deepfake_detector/
├── ai/
│   ├── image_detector/       # CLIP ViT + MLP детектор изображений
│   │   ├── checkpoints/      # Веса обученной модели
│   │   ├── inference.py
│   │   ├── model.py
│   │   └── train.py
│   ├── audio_detector/       # 5× Wav2Vec2 + stacking детектор аудио
│   │   ├── checkpoints/      # Scaler + LogReg + metadata
│   │   └── inference.py
│   └── video_detector/       # VideoMAE детектор видео
│       └── inference.py
├── backend/
│   ├── app.py                # FastAPI endpoints
│   ├── deepfake_runtime.py   # Image service
│   ├── audio_runtime.py      # Audio service
│   └── video_runtime.py      # Video service
├── frontend/
│   └── src/
│       ├── App.vue           # Основной компонент (RU/EN)
│       └── style.css
├── data/                     # Тестовые данные (не в репо)
├── requirements.txt
└── README.md
```

## Особенности

- **Multi-modal**: единый интерфейс для image + audio + video
- **Приватность**: файлы не загружаются на внешние серверы, анализ локальный
- **Ансамбли**: стекинг 5 моделей (аудио), multi-view (изображения), temporal (видео)
- **Двуязычность**: RU / EN интерфейс
- **Async**: неблокирующий FastAPI с thread pool executors для ML inference

---

*Veritas — Hackathon "Антидипфейк: Вызов 2026", IT-Планета*
