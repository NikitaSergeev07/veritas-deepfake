<script setup lang="ts">
import { computed, onMounted, onUnmounted, ref, watch } from "vue";

type Lang = "ru" | "en";
type Tab  = "image" | "audio" | "video";

// ── Image types ──────────────────────────────────────────────────────────────

type ViewScore = {
  name: string;
  fake_probability: number;
  real_probability: number;
};

type ModelScore = {
  model_id: string;
  display_name: string;
  backend: string;
  device: string;
  weight: number;
  fake_probability: number;
  real_probability: number;
  views: ViewScore[];
};

type ImagePredictResult = {
  status: string;
  filename: string;
  label: string;
  confidence: number;
  fake_probability: number;
  real_probability: number;
  summary: string;
  device: string;
  primary_model: string;
  fallback_model?: string | null;
  view_names: string[];
  models: ModelScore[];
  content_type: string;
  extension: string;
  size_bytes: number;
  size_mb: number;
  width: number;
  height: number;
  format: string;
  mode: string;
  orientation: string;
  aspect_ratio: number;
  sha256: string;
  uploaded_at: string;
};

type ImageHealthStatus = {
  status: string;
  service: string;
  version: string;
  max_upload_mb: number;
  supported_extensions: string[];
  supported_content_types: string[];
  device: string;
  model_status: string;
  configured_models: string[];
  loaded_models: string[];
  view_strategy: string[];
  load_errors: Record<string, string>;
};

// ── Audio types ──────────────────────────────────────────────────────────────

type AudioPredictResult = {
  status: string;
  filename: string;
  label: string;
  confidence: number;
  fake_probability: number;
  real_probability: number;
  summary: string;
  device: string;
  primary_model: string;
  base_scores: Record<string, number>;
  size_bytes: number;
  size_mb: number;
  uploaded_at: string;
};

type AudioHealthStatus = {
  status: string;
  service: string;
  device: string;
  model_status: string;
  configured_models: string[];
  loaded_models: string[];
  base_models: string[];
  meta_accuracy: number | null;
  supported_extensions: string[];
  load_errors: Record<string, string>;
};

// ── Video types ─────────────────────────────────────────────────────────────

type VideoSegment = {
  start: number;
  end: number;
  fake_probability: number;
  label: string;
};

type VideoPredictResult = {
  status: string;
  filename: string;
  label: string;
  confidence: number;
  fake_probability: number;
  real_probability: number;
  summary: string;
  device: string;
  primary_model: string;
  duration: number;
  frames_analyzed: number;
  segments: VideoSegment[];
  size_bytes: number;
  size_mb: number;
  uploaded_at: string;
};

type VideoHealthStatus = {
  status: string;
  service: string;
  device: string;
  model_status: string;
  configured_models: string[];
  loaded_models: string[];
  supported_extensions: string[];
  load_errors: Record<string, string>;
};

// ── i18n ─────────────────────────────────────────────────────────────────────

const messages = {
  en: {
    subtitle: "Detect deepfakes in images, audio, and video using on-device AI",
    noDataNote: "Files are not stored · all analysis runs locally on your device",
    tabImage: "Image",
    tabAudio: "Audio",
    tabVideo: "Video",
    dropTitle: "Drop file here",
    dropSub: "or choose a file",
    imgFormats: "JPG, PNG, WEBP, BMP, TIFF · max 10 MB",
    audFormats: "WAV, MP3, FLAC, OGG, M4A · max 50 MB",
    vidFormats: "MP4, MOV, AVI, MKV, WEBM · max 100 MB",
    chooseFile: "Choose file",
    inspect: "Analyze",
    inspecting: "Analyzing…",
    clear: "Clear",
    fileReady: "File ready",
    size: "Size",
    type: "Type",
    resultTitle: "Analysis result",
    labelFake: "Deepfake detected",
    labelReal: "Looks authentic",
    labelUnknown: "Inconclusive",
    confidence: "Confidence",
    previewUnavailable: "Preview unavailable",
    detailsDimensions: "Dimensions",
    detailsFormat: "Format",
    detailsDevice: "Device",
    detailsPrimary: "Model",
    detailsChecksum: "SHA-256",
    detailsUploadedAt: "Processed at",
    probabilityFake: "Fake probability",
    probabilityReal: "Real probability",
    baseScores: "Base model scores",
    segments: "Segment analysis",
    duration: "Duration",
    framesAnalyzed: "Frames analyzed",
    slowNote: "First analysis loads AI models — may take 2–5 min",
    slowNoteVideo: "First video analysis loads the model — may take 1–2 min",
    statusReady: "Ready",
    statusLazy: "Model not yet loaded",
    statusOffline: "Backend offline",
    errors: {
      noFile: "Choose a file first",
      badType: "Unsupported file type",
      tooLarge: "File too large",
      invalidImage: "Cannot read image",
      empty: "Empty file",
      network: "Backend unavailable",
      detector: "Detector loading, please wait",
      unknown: "Something went wrong",
    },
  },
  ru: {
    subtitle: "Определение дипфейков в изображениях, аудио и видео с помощью локальных AI-моделей",
    noDataNote: "Файлы не сохраняются · весь анализ выполняется локально на вашем устройстве",
    tabImage: "Изображение",
    tabAudio: "Аудио",
    tabVideo: "Видео",
    dropTitle: "Перетащите файл сюда",
    dropSub: "или выберите файл",
    imgFormats: "JPG, PNG, WEBP, BMP, TIFF · до 10 МБ",
    audFormats: "WAV, MP3, FLAC, OGG, M4A · до 50 МБ",
    vidFormats: "MP4, MOV, AVI, MKV, WEBM · до 100 МБ",
    chooseFile: "Выбрать файл",
    inspect: "Проверить",
    inspecting: "Проверяем…",
    clear: "Очистить",
    fileReady: "Файл выбран",
    size: "Размер",
    type: "Тип",
    resultTitle: "Результат анализа",
    labelFake: "Обнаружен дипфейк",
    labelReal: "Выглядит подлинным",
    labelUnknown: "Результат неоднозначен",
    confidence: "Уверенность",
    previewUnavailable: "Превью недоступно",
    detailsDimensions: "Размеры",
    detailsFormat: "Формат",
    detailsDevice: "Устройство",
    detailsPrimary: "Модель",
    detailsChecksum: "SHA-256",
    detailsUploadedAt: "Обработано",
    probabilityFake: "Вероятность дипфейка",
    probabilityReal: "Вероятность подлинного",
    baseScores: "Оценки базовых моделей",
    segments: "Анализ по сегментам",
    duration: "Длительность",
    framesAnalyzed: "Кадров проанализировано",
    slowNote: "Первый анализ загружает AI-модели — займёт 2–5 мин",
    slowNoteVideo: "Первый анализ видео загружает модель — займёт 1–2 мин",
    statusReady: "Готово",
    statusLazy: "Модель не загружена",
    statusOffline: "Бэкенд недоступен",
    errors: {
      noFile: "Сначала выберите файл",
      badType: "Неподдерживаемый тип файла",
      tooLarge: "Слишком большой файл",
      invalidImage: "Не удалось прочитать изображение",
      empty: "Пустой файл",
      network: "Бэкенд недоступен",
      detector: "Детектор загружается, подождите",
      unknown: "Что-то пошло не так",
    },
  },
};

const apiUrl = import.meta.env.VITE_API_URL || "http://localhost:8000";

const ALLOWED_IMAGE_TYPES = ["image/jpeg","image/png","image/webp","image/bmp","image/tiff"];
const ALLOWED_IMAGE_EXT   = [".jpg",".jpeg",".png",".webp",".bmp",".tif",".tiff"];
const ALLOWED_AUDIO_EXT   = [".wav",".mp3",".flac",".ogg",".m4a"];
const ALLOWED_VIDEO_EXT   = [".mp4",".mov",".avi",".mkv",".webm"];

// ── State ────────────────────────────────────────────────────────────────────

const lang       = ref<Lang>("ru");
const activeTab  = ref<Tab>("image");

const imgFileInput  = ref<HTMLInputElement | null>(null);
const imgFile       = ref<File | null>(null);
const imgPreviewUrl = ref("");
const imgLoading    = ref(false);
const imgError      = ref("");
const imgResult     = ref<ImagePredictResult | null>(null);
const imgDragActive = ref(false);
const imgHealth     = ref<ImageHealthStatus | null>(null);

const audFileInput  = ref<HTMLInputElement | null>(null);
const audFile       = ref<File | null>(null);
const audLoading    = ref(false);
const audError      = ref("");
const audResult     = ref<AudioPredictResult | null>(null);
const audDragActive = ref(false);
const audHealth     = ref<AudioHealthStatus | null>(null);

const vidFileInput  = ref<HTMLInputElement | null>(null);
const vidFile       = ref<File | null>(null);
const vidLoading    = ref(false);
const vidError      = ref("");
const vidResult     = ref<VideoPredictResult | null>(null);
const vidDragActive = ref(false);
const vidHealth     = ref<VideoHealthStatus | null>(null);

// ── Computed ─────────────────────────────────────────────────────────────────

const ui = computed(() => messages[lang.value]);
const imgMaxMb = computed(() => imgHealth.value?.max_upload_mb ?? 10);

const imgFileMeta = computed(() => {
  if (!imgFile.value) return null;
  return {
    name: imgFile.value.name,
    size: `${(imgFile.value.size / 1024 / 1024).toFixed(2)} MB`,
    type: imgFile.value.type || "image",
  };
});

const audFileMeta = computed(() => {
  if (!audFile.value) return null;
  return {
    name: audFile.value.name,
    size: `${(audFile.value.size / 1024 / 1024).toFixed(2)} MB`,
    type: audFile.value.type || "audio",
  };
});

const vidFileMeta = computed(() => {
  if (!vidFile.value) return null;
  return {
    name: vidFile.value.name,
    size: `${(vidFile.value.size / 1024 / 1024).toFixed(2)} MB`,
    type: vidFile.value.type || "video",
  };
});

const imgResultTone = computed(() => {
  if (!imgResult.value) return "tone-muted";
  if (imgResult.value.label === "fake") return "tone-danger";
  if (imgResult.value.label === "real") return "tone-success";
  return "tone-warning";
});

const audResultTone = computed(() => {
  if (!audResult.value) return "tone-muted";
  if (audResult.value.label === "fake") return "tone-danger";
  if (audResult.value.label === "real") return "tone-success";
  return "tone-warning";
});

const vidResultTone = computed(() => {
  if (!vidResult.value) return "tone-muted";
  if (vidResult.value.label === "fake") return "tone-danger";
  if (vidResult.value.label === "real") return "tone-success";
  return "tone-warning";
});

const imgResultLabel = computed(() => {
  if (!imgResult.value) return "";
  return imgResult.value.label === "fake" ? ui.value.labelFake
       : imgResult.value.label === "real" ? ui.value.labelReal
       : ui.value.labelUnknown;
});

const audResultLabel = computed(() => {
  if (!audResult.value) return "";
  return audResult.value.label === "fake" ? ui.value.labelFake
       : audResult.value.label === "real" ? ui.value.labelReal
       : ui.value.labelUnknown;
});

const vidResultLabel = computed(() => {
  if (!vidResult.value) return "";
  return vidResult.value.label === "fake" ? ui.value.labelFake
       : vidResult.value.label === "real" ? ui.value.labelReal
       : ui.value.labelUnknown;
});

const imgConfidencePct = computed(() => imgResult.value ? Math.round(imgResult.value.confidence * 100) : 0);
const audConfidencePct = computed(() => audResult.value ? Math.round(audResult.value.confidence * 100) : 0);
const vidConfidencePct = computed(() => vidResult.value ? Math.round(vidResult.value.confidence * 100) : 0);

const audBaseScoreRows = computed(() => {
  if (!audResult.value) return [];
  return Object.entries(audResult.value.base_scores).map(([name, score]) => ({
    name,
    pct: Math.round((score as number) * 100),
    score: (score as number).toFixed(3),
    isFake: (score as number) >= 0.5,
  }));
});

const canInspectImg = computed(() => Boolean(imgFile.value) && !imgLoading.value);
const canInspectAud = computed(() => Boolean(audFile.value) && !audLoading.value);
const canInspectVid = computed(() => Boolean(vidFile.value) && !vidLoading.value);

const imgStatusLabel = computed(() => {
  if (!imgHealth.value) return ui.value.statusOffline;
  return imgHealth.value.model_status === "ready" ? ui.value.statusReady : ui.value.statusLazy;
});
const audStatusLabel = computed(() => {
  if (!audHealth.value) return ui.value.statusOffline;
  return audHealth.value.model_status === "ready" ? ui.value.statusReady : ui.value.statusLazy;
});
const vidStatusLabel = computed(() => {
  if (!vidHealth.value) return ui.value.statusOffline;
  return vidHealth.value.model_status === "ready" ? ui.value.statusReady : ui.value.statusLazy;
});
const imgStatusOk = computed(() => imgHealth.value !== null);
const audStatusOk = computed(() => audHealth.value !== null);
const vidStatusOk = computed(() => vidHealth.value !== null);

// ── Lifecycle ────────────────────────────────────────────────────────────────

let _healthTimer: ReturnType<typeof setInterval> | null = null;

onMounted(() => {
  const saved = localStorage.getItem("lang");
  if (saved === "ru" || saved === "en") lang.value = saved as Lang;
  fetchImageHealth();
  fetchAudioHealth();
  fetchVideoHealth();
  _healthTimer = setInterval(() => { fetchImageHealth(); fetchAudioHealth(); fetchVideoHealth(); }, 8000);
});

onUnmounted(() => {
  if (imgPreviewUrl.value) URL.revokeObjectURL(imgPreviewUrl.value);
  if (_healthTimer !== null) clearInterval(_healthTimer);
});

watch(lang, (v) => localStorage.setItem("lang", v));
watch(imgFile, (f) => {
  if (imgPreviewUrl.value) URL.revokeObjectURL(imgPreviewUrl.value);
  imgPreviewUrl.value = f ? URL.createObjectURL(f) : "";
});

// ── Handlers ─────────────────────────────────────────────────────────────────

const mapError = (message: string) => {
  const l = message.toLowerCase();
  if (l.includes("only image") || l.includes("only audio") || l.includes("only video") || l.includes("unsupported")) return ui.value.errors.badType;
  if (l.includes("too large"))    return ui.value.errors.tooLarge;
  if (l.includes("invalid image")) return ui.value.errors.invalidImage;
  if (l.includes("empty"))        return ui.value.errors.empty;
  if (l.includes("detector unavailable") || l.includes("audio detector") || l.includes("video detector")) return ui.value.errors.detector;
  if (l.includes("failed to fetch") || l.includes("network")) return ui.value.errors.network;
  return ui.value.errors.unknown;
};

const setImgFile = (file?: File | null) => {
  if (!file) return;
  const extOk = ALLOWED_IMAGE_EXT.some((e) => file.name.toLowerCase().endsWith(e));
  if (!ALLOWED_IMAGE_TYPES.includes(file.type) && !extOk) { imgError.value = ui.value.errors.badType; return; }
  if (file.size > imgMaxMb.value * 1024 * 1024) { imgError.value = ui.value.errors.tooLarge; return; }
  imgFile.value = file; imgResult.value = null; imgError.value = "";
};

const clearImgFile = () => {
  imgFile.value = null; imgResult.value = null; imgError.value = "";
  if (imgFileInput.value) imgFileInput.value.value = "";
};

const onImgDrop     = (e: DragEvent) => { e.preventDefault(); imgDragActive.value = false; setImgFile(e.dataTransfer?.files?.[0]); };
const onImgDragOver = (e: DragEvent) => { e.preventDefault(); imgDragActive.value = true; };
const onImgDragLeave = () => { imgDragActive.value = false; };

const inspectImage = async () => {
  if (!imgFile.value) { imgError.value = ui.value.errors.noFile; return; }
  imgLoading.value = true; imgError.value = ""; imgResult.value = null;
  try {
    const form = new FormData();
    form.append("file", imgFile.value);
    const resp = await fetch(`${apiUrl}/predict`, { method: "POST", body: form });
    if (!resp.ok) { const p = await resp.json().catch(() => null); throw new Error(p?.detail || resp.statusText); }
    imgResult.value = await resp.json() as ImagePredictResult;
    fetchImageHealth();
  } catch (err) {
    imgError.value = mapError(err instanceof Error ? err.message : String(err));
    fetchImageHealth();
  } finally {
    imgLoading.value = false;
  }
};

const fetchImageHealth = async () => {
  try {
    const r = await fetch(`${apiUrl}/health`);
    imgHealth.value = r.ok ? await r.json() : null;
  } catch { imgHealth.value = null; }
};

const setAudFile = (file?: File | null) => {
  if (!file) return;
  const extOk = ALLOWED_AUDIO_EXT.some((e) => file.name.toLowerCase().endsWith(e));
  if (!extOk) { audError.value = ui.value.errors.badType; return; }
  if (file.size > 50 * 1024 * 1024) { audError.value = ui.value.errors.tooLarge; return; }
  audFile.value = file; audResult.value = null; audError.value = "";
};

const clearAudFile = () => {
  audFile.value = null; audResult.value = null; audError.value = "";
  if (audFileInput.value) audFileInput.value.value = "";
};

const onAudDrop     = (e: DragEvent) => { e.preventDefault(); audDragActive.value = false; setAudFile(e.dataTransfer?.files?.[0]); };
const onAudDragOver = (e: DragEvent) => { e.preventDefault(); audDragActive.value = true; };
const onAudDragLeave = () => { audDragActive.value = false; };

const inspectAudio = async () => {
  if (!audFile.value) { audError.value = ui.value.errors.noFile; return; }
  audLoading.value = true; audError.value = ""; audResult.value = null;
  try {
    const form = new FormData();
    form.append("file", audFile.value);
    const resp = await fetch(`${apiUrl}/predict/audio`, { method: "POST", body: form });
    if (!resp.ok) { const p = await resp.json().catch(() => null); throw new Error(p?.detail || resp.statusText); }
    audResult.value = await resp.json() as AudioPredictResult;
    fetchAudioHealth();
  } catch (err) {
    audError.value = mapError(err instanceof Error ? err.message : String(err));
    fetchAudioHealth();
  } finally {
    audLoading.value = false;
  }
};

const fetchAudioHealth = async () => {
  try {
    const r = await fetch(`${apiUrl}/health/audio`);
    audHealth.value = r.ok ? await r.json() : null;
  } catch { audHealth.value = null; }
};

const setVidFile = (file?: File | null) => {
  if (!file) return;
  const extOk = ALLOWED_VIDEO_EXT.some((e) => file.name.toLowerCase().endsWith(e));
  if (!extOk) { vidError.value = ui.value.errors.badType; return; }
  if (file.size > 100 * 1024 * 1024) { vidError.value = ui.value.errors.tooLarge; return; }
  vidFile.value = file; vidResult.value = null; vidError.value = "";
};

const clearVidFile = () => {
  vidFile.value = null; vidResult.value = null; vidError.value = "";
  if (vidFileInput.value) vidFileInput.value.value = "";
};

const onVidDrop     = (e: DragEvent) => { e.preventDefault(); vidDragActive.value = false; setVidFile(e.dataTransfer?.files?.[0]); };
const onVidDragOver = (e: DragEvent) => { e.preventDefault(); vidDragActive.value = true; };
const onVidDragLeave = () => { vidDragActive.value = false; };

const inspectVideo = async () => {
  if (!vidFile.value) { vidError.value = ui.value.errors.noFile; return; }
  vidLoading.value = true; vidError.value = ""; vidResult.value = null;
  try {
    const form = new FormData();
    form.append("file", vidFile.value);
    const resp = await fetch(`${apiUrl}/predict/video`, { method: "POST", body: form });
    if (!resp.ok) { const p = await resp.json().catch(() => null); throw new Error(p?.detail || resp.statusText); }
    vidResult.value = await resp.json() as VideoPredictResult;
    fetchVideoHealth();
  } catch (err) {
    vidError.value = mapError(err instanceof Error ? err.message : String(err));
    fetchVideoHealth();
  } finally {
    vidLoading.value = false;
  }
};

const fetchVideoHealth = async () => {
  try {
    const r = await fetch(`${apiUrl}/health/video`);
    vidHealth.value = r.ok ? await r.json() : null;
  } catch { vidHealth.value = null; }
};

const setLanguage = (v: Lang) => { lang.value = v; };
</script>

<template>
  <div class="page">

    <!-- ── Topbar ──────────────────────────────────────────────────────────── -->
    <header class="topbar">
      <div class="brand">
        <div class="brand-mark">
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
            <path d="M4 6.5L12 2.5L20 6.5V17.5L12 21.5L4 17.5V6.5Z" stroke="rgba(12,20,34,0.9)" stroke-width="1.4"/>
            <path d="M8 8.5L12 6.5L16 8.5V15.5L12 17.5L8 15.5V8.5Z" fill="rgba(8,17,31,0.75)"/>
          </svg>
        </div>
        <span class="brand-name">Veritas</span>
      </div>
      <div class="lang-toggle">
        <button class="lang-btn" :class="{ active: lang === 'ru' }" @click="setLanguage('ru')">RU</button>
        <button class="lang-btn" :class="{ active: lang === 'en' }" @click="setLanguage('en')">EN</button>
      </div>
    </header>

    <!-- ── Hero upload ─────────────────────────────────────────────────────── -->
    <main class="main">
      <div class="vt-center">

        <!-- Logo + subtitle -->
        <div class="vt-brand">
          <div class="vt-logo-mark">
            <svg width="36" height="36" viewBox="0 0 24 24" fill="none">
              <path d="M4 6.5L12 2.5L20 6.5V17.5L12 21.5L4 17.5V6.5Z" stroke="rgba(12,20,34,0.9)" stroke-width="1.4"/>
              <path d="M8 8.5L12 6.5L16 8.5V15.5L12 17.5L8 15.5V8.5Z" fill="rgba(8,17,31,0.6)"/>
            </svg>
          </div>
          <h1 class="vt-title">Veritas</h1>
        </div>
        <p class="vt-subtitle">{{ ui.subtitle }}</p>

        <!-- Tab switcher -->
        <nav class="vt-tabs">
          <button class="vt-tab" :class="{ active: activeTab === 'image' }" @click="activeTab = 'image'">
            <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <rect x="3" y="3" width="18" height="18" rx="2"/><circle cx="8.5" cy="8.5" r="1.5"/>
              <path d="M21 15l-5-5L5 21"/>
            </svg>
            {{ ui.tabImage }}
          </button>
          <button class="vt-tab" :class="{ active: activeTab === 'audio' }" @click="activeTab = 'audio'">
            <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <path d="M9 18V5l12-2v13"/><circle cx="6" cy="18" r="3"/><circle cx="18" cy="16" r="3"/>
            </svg>
            {{ ui.tabAudio }}
          </button>
          <button class="vt-tab" :class="{ active: activeTab === 'video' }" @click="activeTab = 'video'">
            <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <polygon points="5 3 19 12 5 21 5 3"/>
            </svg>
            {{ ui.tabVideo }}
          </button>
          <div class="vt-tab-line"></div>
        </nav>

        <!-- Upload panel -->
        <div class="vt-panel">

          <!-- ── Image tab ── -->
          <template v-if="activeTab === 'image'">
            <div class="vt-drop" :class="{ dragging: imgDragActive }"
              @click="imgFileInput?.click()" @drop="onImgDrop" @dragover="onImgDragOver" @dragleave="onImgDragLeave">
              <div class="vt-drop-icon">
                <svg width="52" height="52" viewBox="0 0 52 52" fill="none">
                  <rect x="14" y="8" width="24" height="30" rx="3" stroke="currentColor" stroke-width="1.5"/>
                  <path d="M14 30l8-8 6 6 4-4 4 6" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
                  <circle cx="22" cy="18" r="2" fill="currentColor"/>
                  <path d="M26 44v-8M22 40l4-4 4 4" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
                </svg>
              </div>
              <button class="vt-choose-btn" @click.stop="imgFileInput?.click()">{{ ui.chooseFile }}</button>
              <p class="vt-formats">{{ ui.imgFormats }}</p>
            </div>
            <input ref="imgFileInput" class="file-input" type="file" accept="image/*"
              @change="setImgFile(($event.target as HTMLInputElement).files?.[0])"/>

            <div v-if="imgFileMeta" class="vt-file-row">
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="var(--accent-2)" stroke-width="2">
                <path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z"/><path d="M14 2v6h6"/>
              </svg>
              <span class="vt-fname">{{ imgFileMeta.name }}</span>
              <span class="vt-fsize">{{ imgFileMeta.size }}</span>
              <button class="vt-clear-btn" @click="clearImgFile">×</button>
            </div>

            <div class="vt-actions" v-if="imgFileMeta">
              <button class="btn-analyze" :disabled="!canInspectImg" @click="inspectImage">
                <span v-if="imgLoading" class="spinner"></span>
                {{ imgLoading ? ui.inspecting : ui.inspect }}
              </button>
            </div>

            <div v-if="imgError" class="vt-error">{{ imgError }}</div>
          </template>

          <!-- ── Audio tab ── -->
          <template v-if="activeTab === 'audio'">
            <div class="vt-drop" :class="{ dragging: audDragActive }"
              @click="audFileInput?.click()" @drop="onAudDrop" @dragover="onAudDragOver" @dragleave="onAudDragLeave">
              <div class="vt-drop-icon">
                <svg width="52" height="52" viewBox="0 0 52 52" fill="none">
                  <path d="M20 38V18l20-4v20" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
                  <circle cx="14" cy="38" r="6" stroke="currentColor" stroke-width="1.5"/>
                  <circle cx="34" cy="34" r="6" stroke="currentColor" stroke-width="1.5"/>
                </svg>
              </div>
              <button class="vt-choose-btn" @click.stop="audFileInput?.click()">{{ ui.chooseFile }}</button>
              <p class="vt-formats">{{ ui.audFormats }}</p>
            </div>
            <input ref="audFileInput" class="file-input" type="file" accept="audio/*,.wav,.mp3,.flac,.ogg,.m4a"
              @change="setAudFile(($event.target as HTMLInputElement).files?.[0])"/>

            <div v-if="audFileMeta" class="vt-file-row">
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="var(--accent-2)" stroke-width="2">
                <path d="M9 18V5l12-2v13"/><circle cx="6" cy="18" r="3"/><circle cx="18" cy="16" r="3"/>
              </svg>
              <span class="vt-fname">{{ audFileMeta.name }}</span>
              <span class="vt-fsize">{{ audFileMeta.size }}</span>
              <button class="vt-clear-btn" @click="clearAudFile">×</button>
            </div>

            <div class="vt-actions" v-if="audFileMeta">
              <button class="btn-analyze" :disabled="!canInspectAud" @click="inspectAudio">
                <span v-if="audLoading" class="spinner"></span>
                {{ audLoading ? ui.inspecting : ui.inspect }}
              </button>
            </div>

            <p v-if="!audHealth || audHealth.model_status !== 'ready'" class="vt-slow-note">
              ⏱ {{ ui.slowNote }}
            </p>
            <div v-if="audError" class="vt-error">{{ audError }}</div>
          </template>

          <!-- ── Video tab ── -->
          <template v-if="activeTab === 'video'">
            <div class="vt-drop" :class="{ dragging: vidDragActive }"
              @click="vidFileInput?.click()" @drop="onVidDrop" @dragover="onVidDragOver" @dragleave="onVidDragLeave">
              <div class="vt-drop-icon">
                <svg width="52" height="52" viewBox="0 0 52 52" fill="none">
                  <rect x="10" y="12" width="32" height="22" rx="3" stroke="currentColor" stroke-width="1.5"/>
                  <polygon points="22,18 34,23 22,28" fill="currentColor" opacity="0.6"/>
                  <path d="M26 44v-8M22 40l4-4 4 4" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
                </svg>
              </div>
              <button class="vt-choose-btn" @click.stop="vidFileInput?.click()">{{ ui.chooseFile }}</button>
              <p class="vt-formats">{{ ui.vidFormats }}</p>
            </div>
            <input ref="vidFileInput" class="file-input" type="file" accept="video/*,.mp4,.mov,.avi,.mkv,.webm"
              @change="setVidFile(($event.target as HTMLInputElement).files?.[0])"/>

            <div v-if="vidFileMeta" class="vt-file-row">
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="var(--accent-2)" stroke-width="2">
                <polygon points="5 3 19 12 5 21 5 3"/>
              </svg>
              <span class="vt-fname">{{ vidFileMeta.name }}</span>
              <span class="vt-fsize">{{ vidFileMeta.size }}</span>
              <button class="vt-clear-btn" @click="clearVidFile">×</button>
            </div>

            <div class="vt-actions" v-if="vidFileMeta">
              <button class="btn-analyze" :disabled="!canInspectVid" @click="inspectVideo">
                <span v-if="vidLoading" class="spinner"></span>
                {{ vidLoading ? ui.inspecting : ui.inspect }}
              </button>
            </div>

            <p v-if="!vidHealth || vidHealth.model_status !== 'ready'" class="vt-slow-note">
              ⏱ {{ ui.slowNoteVideo }}
            </p>
            <div v-if="vidError" class="vt-error">{{ vidError }}</div>
          </template>

        </div>

        <!-- Status + note -->
        <div class="vt-footer-row">
          <div class="vt-status">
            <span class="status-dot"
              :class="activeTab === 'image' ? (imgStatusOk ? 'dot-ok' : 'dot-err')
                    : activeTab === 'audio' ? (audStatusOk ? 'dot-ok' : 'dot-err')
                    : (vidStatusOk ? 'dot-ok' : 'dot-err')">
            </span>
            <span class="status-text">
              {{ activeTab === 'image' ? imgStatusLabel : activeTab === 'audio' ? audStatusLabel : vidStatusLabel }}
            </span>
          </div>
          <span class="vt-note">{{ ui.noDataNote }}</span>
        </div>

      </div>
    </main>

    <!-- ── Image result ────────────────────────────────────────────────────── -->
    <section v-if="imgResult && activeTab === 'image'" class="results container">
      <div class="res-verdict" :class="imgResultTone">
        <div class="res-label">{{ imgResultLabel }}</div>
        <div class="res-conf">{{ ui.confidence }}: {{ imgConfidencePct }}%</div>
      </div>

      <div class="res-grid">
        <!-- Preview -->
        <div class="res-card">
          <div class="preview-frame">
            <img v-if="imgPreviewUrl" :src="imgPreviewUrl" :alt="imgResult.filename" class="preview-img"/>
            <div v-else class="preview-empty">{{ ui.previewUnavailable }}</div>
          </div>
        </div>

        <!-- Probabilities -->
        <div class="res-card">
          <div class="res-section-title">{{ ui.probabilityFake }}</div>
          <div class="big-prob" :class="imgResultTone">{{ Math.round(imgResult.fake_probability * 100) }}%</div>
          <div class="prob-bar-track">
            <div class="prob-bar-fill"
              :style="{ width: `${imgResult.fake_probability * 100}%`,
                        background: imgResult.fake_probability >= 0.5 ? 'var(--danger)' : 'var(--success)' }"/>
          </div>

          <div class="res-meta-grid">
            <div class="res-meta-item">
              <span class="res-meta-label">{{ ui.detailsDimensions }}</span>
              <span class="mono">{{ imgResult.width }}×{{ imgResult.height }}</span>
            </div>
            <div class="res-meta-item">
              <span class="res-meta-label">{{ ui.detailsFormat }}</span>
              <span>{{ imgResult.format }}</span>
            </div>
            <div class="res-meta-item">
              <span class="res-meta-label">{{ ui.detailsDevice }}</span>
              <span class="mono">{{ imgResult.device }}</span>
            </div>
            <div class="res-meta-item">
              <span class="res-meta-label">{{ ui.detailsPrimary }}</span>
              <span class="mono">{{ imgResult.primary_model }}</span>
            </div>
            <div class="res-meta-item">
              <span class="res-meta-label">{{ ui.detailsUploadedAt }}</span>
              <span>{{ new Date(imgResult.uploaded_at).toLocaleTimeString() }}</span>
            </div>
            <div class="res-meta-item" style="grid-column: 1/-1">
              <span class="res-meta-label">{{ ui.detailsChecksum }}</span>
              <span class="mono hash">{{ imgResult.sha256 }}</span>
            </div>
          </div>

          <!-- Per-view breakdown -->
          <div v-for="model in imgResult.models" :key="model.model_id" class="view-breakdown">
            <div v-for="view in model.views" :key="view.name" class="view-row">
              <span class="view-name">{{ view.name }}</span>
              <div class="prob-bar-track sm">
                <div class="prob-bar-fill"
                  :style="{ width: `${view.fake_probability * 100}%`,
                            background: view.fake_probability >= 0.5 ? 'var(--danger)' : 'var(--success)' }"/>
              </div>
              <span class="view-score mono" :class="view.fake_probability >= 0.5 ? 'tone-danger' : 'tone-success'">
                {{ (view.fake_probability * 100).toFixed(1) }}%
              </span>
            </div>
          </div>
        </div>
      </div>
    </section>

    <!-- ── Audio result ────────────────────────────────────────────────────── -->
    <section v-if="audResult && activeTab === 'audio'" class="results container">
      <div class="res-verdict" :class="audResultTone">
        <div class="res-label">{{ audResultLabel }}</div>
        <div class="res-conf">{{ ui.confidence }}: {{ audConfidencePct }}%</div>
      </div>

      <div class="res-grid">
        <!-- Audio icon card -->
        <div class="res-card audio-card">
          <svg width="56" height="56" viewBox="0 0 24 24" fill="none" stroke="var(--accent-2)" stroke-width="1.4">
            <path d="M9 18V5l12-2v13" stroke-linecap="round" stroke-linejoin="round"/>
            <circle cx="6" cy="18" r="3"/><circle cx="18" cy="16" r="3"/>
          </svg>
          <div class="audio-fname">{{ audResult.filename }}</div>
          <div class="audio-size">{{ audResult.size_mb.toFixed(2) }} MB</div>
          <div class="big-prob" :class="audResultTone">{{ audConfidencePct }}%</div>
          <div class="prob-bar-track">
            <div class="prob-bar-fill"
              :style="{ width: `${audResult.fake_probability * 100}%`,
                        background: audResult.fake_probability >= 0.5 ? 'var(--danger)' : 'var(--success)' }"/>
          </div>
          <div class="res-meta-grid" style="margin-top:16px">
            <div class="res-meta-item">
              <span class="res-meta-label">{{ ui.detailsDevice }}</span>
              <span class="mono">{{ audResult.device }}</span>
            </div>
            <div class="res-meta-item">
              <span class="res-meta-label">{{ ui.detailsPrimary }}</span>
              <span class="mono">{{ audResult.primary_model }}</span>
            </div>
          </div>
        </div>

        <!-- Base model scores -->
        <div class="res-card">
          <div class="res-section-title">{{ ui.baseScores }}</div>
          <div class="base-scores">
            <div v-for="row in audBaseScoreRows" :key="row.name" class="base-row">
              <span class="base-name mono">{{ row.name }}</span>
              <div class="prob-bar-track sm">
                <div class="prob-bar-fill"
                  :style="{ width: `${row.pct}%`,
                            background: row.isFake ? 'var(--danger)' : 'var(--success)' }"/>
              </div>
              <span class="base-score mono" :class="row.isFake ? 'tone-danger' : 'tone-success'">{{ row.pct }}%</span>
            </div>
          </div>
        </div>
      </div>
    </section>

    <!-- ── Video result ───────────────────────────────────────────────────── -->
    <section v-if="vidResult && activeTab === 'video'" class="results container">
      <div class="res-verdict" :class="vidResultTone">
        <div class="res-label">{{ vidResultLabel }}</div>
        <div class="res-conf">{{ ui.confidence }}: {{ vidConfidencePct }}%</div>
      </div>

      <div class="res-grid">
        <!-- Video info card -->
        <div class="res-card audio-card">
          <svg width="56" height="56" viewBox="0 0 24 24" fill="none" stroke="var(--accent-2)" stroke-width="1.4">
            <rect x="2" y="4" width="20" height="14" rx="2"/>
            <polygon points="9.5,8 16,11 9.5,14" fill="var(--accent-2)" opacity="0.5"/>
          </svg>
          <div class="audio-fname">{{ vidResult.filename }}</div>
          <div class="audio-size">{{ vidResult.size_mb.toFixed(2) }} MB</div>
          <div class="big-prob" :class="vidResultTone">{{ vidConfidencePct }}%</div>
          <div class="prob-bar-track">
            <div class="prob-bar-fill"
              :style="{ width: `${vidResult.fake_probability * 100}%`,
                        background: vidResult.fake_probability >= 0.5 ? 'var(--danger)' : 'var(--success)' }"/>
          </div>
          <div class="res-meta-grid" style="margin-top:16px">
            <div class="res-meta-item">
              <span class="res-meta-label">{{ ui.duration }}</span>
              <span class="mono">{{ vidResult.duration }}s</span>
            </div>
            <div class="res-meta-item">
              <span class="res-meta-label">{{ ui.framesAnalyzed }}</span>
              <span class="mono">{{ vidResult.frames_analyzed }}</span>
            </div>
            <div class="res-meta-item">
              <span class="res-meta-label">{{ ui.detailsDevice }}</span>
              <span class="mono">{{ vidResult.device }}</span>
            </div>
            <div class="res-meta-item">
              <span class="res-meta-label">{{ ui.detailsPrimary }}</span>
              <span class="mono">{{ vidResult.primary_model }}</span>
            </div>
          </div>
        </div>

        <!-- Segments card -->
        <div class="res-card">
          <div class="res-section-title">{{ ui.segments }}</div>

          <!-- Timeline bar -->
          <div v-if="vidResult.segments.length" class="seg-timeline">
            <div v-for="(seg, i) in vidResult.segments" :key="i"
              class="seg-block"
              :class="seg.label === 'fake' ? 'seg-fake' : seg.label === 'real' ? 'seg-real' : 'seg-uncertain'"
              :style="{ width: `${(seg.end - seg.start) / vidResult.duration * 100}%` }">
              <div class="seg-tooltip">
                <span class="seg-tooltip-time">{{ seg.start }}s – {{ seg.end }}s</span>
                <span class="seg-tooltip-label"
                  :class="seg.label === 'fake' ? 'tone-danger' : seg.label === 'real' ? 'tone-success' : 'tone-warning'">
                  {{ seg.label === 'fake'
                    ? (lang === 'ru' ? 'Дипфейк' : 'Deepfake')
                    : seg.label === 'real'
                      ? (lang === 'ru' ? 'Подлинное' : 'Authentic')
                      : (lang === 'ru' ? 'Неоднозначно' : 'Uncertain') }}
                </span>
                <span class="seg-tooltip-prob">{{ Math.round(seg.fake_probability * 100) }}% {{ lang === 'ru' ? 'вероятность подделки' : 'fake probability' }}</span>
              </div>
            </div>
          </div>

          <!-- Segment details -->
          <div v-if="vidResult.segments.length" class="base-scores">
            <div v-for="(seg, i) in vidResult.segments" :key="i" class="base-row">
              <span class="base-name mono">
                <span class="seg-index">#{{ i + 1 }}</span>
                {{ seg.start }}s – {{ seg.end }}s
              </span>
              <div class="prob-bar-track sm">
                <div class="prob-bar-fill"
                  :style="{ width: `${seg.fake_probability * 100}%`,
                            background: seg.fake_probability >= 0.5 ? 'var(--danger)' : 'var(--success)' }"/>
              </div>
              <span class="base-score mono"
                :class="seg.label === 'fake' ? 'tone-danger' : seg.label === 'real' ? 'tone-success' : 'tone-warning'">
                {{ seg.label === 'fake'
                  ? (lang === 'ru' ? 'Фейк' : 'Fake')
                  : seg.label === 'real'
                    ? (lang === 'ru' ? 'Ок' : 'Real')
                    : (lang === 'ru' ? 'Н/о' : 'N/a') }}
                {{ Math.round(seg.fake_probability * 100) }}%
              </span>
            </div>
          </div>

          <p v-if="!vidResult.segments.length" class="vt-slow-note" style="margin:16px 0 0">
            {{ lang === 'ru' ? 'Видео слишком короткое для сегментного анализа' : 'Video too short for segment analysis' }}
          </p>
        </div>
      </div>
    </section>

    <footer class="footer">
      <span>Veritas · Hackathon 2026</span>
    </footer>

  </div>
</template>
