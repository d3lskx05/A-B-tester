import os
import time
import re
import math
import psutil
import numpy as np
import pandas as pd
from io import BytesIO
from typing import List, Tuple, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import streamlit as st
from sentence_transformers import SentenceTransformer

# =========================================
# ============== НАСТРОЙКИ ================
# =========================================

st.set_page_config(page_title="A/B-тестер эмбеддинговых моделей", layout="wide")

DEFAULT_DATASET_PATH = "/mnt/data/data6.xlsx"  # твой загруженный файл
DEFAULT_TOP_K = 5

# Предустановленные модели для удобства (можно вводить любые HF id вручную)
PRESET_MODELS = [
    "intfloat/multilingual-e5-small",
    "intfloat/multilingual-e5-base",
    "BAAI/bge-m3",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "sentence-transformers/all-MiniLM-L6-v2",
]

# Правила префиксов по семействам (по умолчанию; можно переопределять в UI)
# role: "query" | "passage"
FAMILY_PREFIX_RULES = [
    {
        "pattern": r"(^|/)intfloat/.*e5.*",
        "query_prefix": "query: ",
        "passage_prefix": "passage: ",
    },
    {
        "pattern": r"(^|/)BAAI/bge-.*",
        "query_prefix": "query: ",
        "passage_prefix": "document: ",
    },
    # Универсальные (MiniLM/mpnet/…): без префиксов
]

# =========================================
# =========== КЕШИ / ВСПОМОГАТЕЛЬНЫЕ ======
# =========================================

@st.cache_resource(show_spinner=False)
def get_model_cached(model_name: str) -> SentenceTransformer:
    return SentenceTransformer(model_name)

def detect_family_prefixes(model_name: str) -> Tuple[Optional[str], Optional[str]]:
    for rule in FAMILY_PREFIX_RULES:
        if re.search(rule["pattern"], model_name):
            return rule["query_prefix"], rule["passage_prefix"]
    return None, None  # по умолчанию префиксов нет

def maybe_prefix(texts: List[str], prefix: Optional[str], add_prefix: bool) -> List[str]:
    if add_prefix and prefix:
        return [prefix + t for t in texts]
    return texts

def cosine_sim_matrix(vec: np.ndarray, mat: np.ndarray, mat_norms: np.ndarray) -> np.ndarray:
    # vec: (d,), mat: (N, d)
    vnorm = np.linalg.norm(vec) or 1e-10
    return (mat @ vec) / (mat_norms * vnorm)

def human_bytes(n_bytes: int) -> str:
    if n_bytes < 1024: return f"{n_bytes} B"
    for unit in ["KB","MB","GB","TB"]:
        n_bytes /= 1024.0
        if n_bytes < 1024.0:
            return f"{n_bytes:.1f} {unit}"
    return f"{n_bytes:.1f} PB"

def get_process_mem():
    proc = psutil.Process(os.getpid())
    rss = proc.memory_info().rss
    return rss

# =========================================
# ============ ЗАГРУЗКА ДАННЫХ ============
# =========================================

@st.cache_data(show_spinner=False)
def load_excel_any(path_or_bytes: bytes | str) -> pd.DataFrame:
    if isinstance(path_or_bytes, (bytes, bytearray)):
        df = pd.read_excel(BytesIO(path_or_bytes))
    else:
        df = pd.read_excel(path_or_bytes)
    # Ожидаемые колонки: phrase, topics*, comment (как в твоём проекте)
    # Собираем topics-колонки:
    topic_cols = [c for c in df.columns if str(c).lower().startswith("topics")]
    if not topic_cols:
        # Если нет topics-колонок — создадим пустые
        df["topics"] = [[] for _ in range(len(df))]
    else:
        df["topics"] = df[topic_cols].astype(str).agg(
            lambda x: [v for v in x if v and v != "nan"], axis=1
        )
    if "phrase" not in df.columns:
        raise ValueError("В Excel не найдена колонка 'phrase'")
    if "comment" not in df.columns:
        df["comment"] = ""

    # Нормализованные поля
    df["phrase_full"] = df["phrase"]
    df["phrase_proc"] = df["phrase"].astype(str).str.lower().str.replace(r"\s+", " ", regex=True).str.strip()
    return df[["phrase", "phrase_proc", "phrase_full", "topics", "comment"]]

# =========================================
# ====== ПОДГОТОВКА ЭМБЕДДИНГОВ БАЗЫ ======
# =========================================

@st.cache_data(show_spinner=False)
def compute_phrase_embeddings(
    df: pd.DataFrame,
    model_name: str,
    add_prefix: bool,
    custom_query_prefix: str | None,
    custom_passage_prefix: str | None,
    batch_size: int = 128,
) -> Dict[str, np.ndarray]:
    """
    Предвычисляет эмбеддинги для базы фраз под конкретную модель и флаги префиксов.
    Возвращает dict: {"embeddings": (N, d), "norms": (N,), "dim": d}
    """
    model = get_model_cached(model_name)

    # Автодетект префиксов + кастомные переопределения из UI
    auto_q, auto_p = detect_family_prefixes(model_name)
    q_prefix = (custom_query_prefix if (custom_query_prefix is not None) else auto_q)
    p_prefix = (custom_passage_prefix if (custom_passage_prefix is not None) else auto_p)

    phrases = df["phrase_proc"].tolist()
    passages = maybe_prefix(phrases, p_prefix, add_prefix)

    embs: List[np.ndarray] = []
    for i in range(0, len(passages), batch_size):
        batch = passages[i:i+batch_size]
        batch_embs = model.encode(batch, convert_to_numpy=True, show_progress_bar=False)
        embs.append(batch_embs.astype("float32"))

    if embs:
        emb_mat = np.vstack(embs)
        norms = np.linalg.norm(emb_mat, axis=1)
        norms[norms == 0] = 1e-10
    else:
        dim = model.get_sentence_embedding_dimension()
        emb_mat = np.zeros((0, dim), dtype="float32")
        norms = np.zeros((0,), dtype="float32")

    return {
        "embeddings": emb_mat,
        "norms": norms,
        "dim": emb_mat.shape[1] if emb_mat.size else model.get_sentence_embedding_dimension(),
    }

# =========================================
# =============== ПОИСК ====================
# =========================================

def search_topk(
    query_text: str,
    df: pd.DataFrame,
    model_name: str,
    precomputed: Dict[str, np.ndarray],
    add_prefix: bool,
    custom_query_prefix: str | None,
    custom_passage_prefix: str | None,
    top_k: int = DEFAULT_TOP_K,
    hybrid_query: bool = True,
) -> Dict:
    """
    Возвращает результаты поиска + метрики.
    hybrid_query=True: как в твоём проекте — смешиваем query с префиксом и без.
    """
    model = get_model_cached(model_name)

    auto_q, auto_p = detect_family_prefixes(model_name)
    q_prefix = (custom_query_prefix if (custom_query_prefix is not None) else auto_q)

    phrase_embs = precomputed["embeddings"]
    phrase_norms = precomputed["norms"]

    if phrase_embs is None or phrase_embs.size == 0:
        return {"results": [], "latency_s": 0.0}

    t0 = time.time()
    # Запросы: с префиксом и без (если нужно)
    q_proc = str(query_text).lower().strip()
    q_pref = maybe_prefix([q_proc], q_prefix, add_prefix)[0] if q_prefix else q_proc

    q_emb_pref = model.encode(q_pref, convert_to_numpy=True, show_progress_bar=False).astype("float32")
    sims_pref = cosine_sim_matrix(q_emb_pref, phrase_embs, phrase_norms)

    if hybrid_query:
        q_emb_raw = model.encode(q_proc, convert_to_numpy=True, show_progress_bar=False).astype("float32")
        sims_raw = cosine_sim_matrix(q_emb_raw, phrase_embs, phrase_norms)
        sims = (sims_pref + sims_raw) / 2.0
        sims = np.nan_to_num(sims, neginf=0.0, posinf=0.0)
    else:
        sims = sims_pref

    idx = np.argsort(sims)[::-1][:top_k]
    results = [
        {
            "score": float(sims[i]),
            "phrase_full": df.iloc[i]["phrase_full"],
            "topics": df.iloc[i]["topics"],
            "comment": df.iloc[i]["comment"],
        }
        for i in idx
    ]
    latency = time.time() - t0

    return {"results": results, "latency_s": latency}

# =========================================
# ========= СИМУЛЯЦИЯ НАГРУЗКИ ============
# =========================================

def simulate_users(
    n_users: int,
    n_requests_per_user: int,
    query_text: str,
    df: pd.DataFrame,
    model_name: str,
    precomputed: Dict[str, np.ndarray],
    add_prefix: bool,
    custom_query_prefix: str | None,
    custom_passage_prefix: str | None,
    top_k: int,
) -> Dict:
    """
    Простейшая параллельная симуляция «виртуальных пользователей» потоками.
    Измеряем latency на каждый запрос.
    """
    latencies = []
    start_rss = get_process_mem()
    start_cpu = psutil.cpu_percent(interval=None)
    t0 = time.time()

    def one_call():
        r = search_topk(
            query_text=query_text,
            df=df,
            model_name=model_name,
            precomputed=precomputed,
            add_prefix=add_prefix,
            custom_query_prefix=custom_query_prefix,
            custom_passage_prefix=custom_passage_prefix,
            top_k=top_k,
        )
        return r["latency_s"]

    with ThreadPoolExecutor(max_workers=n_users) as ex:
        futures = []
        for _ in range(n_users * n_requests_per_user):
            futures.append(ex.submit(one_call))
        for fut in as_completed(futures):
            latencies.append(fut.result())

    total_time = time.time() - t0
    end_rss = get_process_mem()
    end_cpu = psutil.cpu_percent(interval=None)

    latencies = np.array(latencies) if latencies else np.array([0.0])

    return {
        "count": int(len(latencies)),
        "avg_ms": float(latencies.mean() * 1000),
        "p95_ms": float(np.percentile(latencies, 95) * 1000),
        "min_ms": float(latencies.min() * 1000),
        "max_ms": float(latencies.max() * 1000),
        "total_time_s": float(total_time),
        "cpu_start_pct": float(start_cpu),
        "cpu_end_pct": float(end_cpu),
        "mem_start": int(start_rss),
        "mem_end": int(end_rss),
    }

# =========================================
# =============== UI ======================
# =========================================

st.title("🔬 Универсальный A/B-тестер моделей эмбеддингов")
st.caption("Сравнение качества, скорости и ресурсов для E5/BGE/MiniLM/mpnet и др.")

# ---- Сайдбар: Загрузка / Метрики окружения / Настройки логгирования
with st.sidebar:
    st.header("⚙️ Настройки")

    st.subheader("📥 Датасет")
    uploaded = st.file_uploader("Загрузить Excel (.xlsx) с колонкой 'phrase'", type=["xlsx"])
    if uploaded is not None:
        df = load_excel_any(uploaded.read())
        st.success(f"Загружено строк: {len(df)}")
    else:
        # Фоллбэк на заранее загруженный файл
        df = load_excel_any(DEFAULT_DATASET_PATH)
        st.info(f"Используем {DEFAULT_DATASET_PATH}. Строк: {len(df)}")

    st.subheader("🧪 Лёгкие метрики датасета")
    n_phrases = len(df)
    all_topics = sorted({t for row in df["topics"] for t in row}) if "topics" in df.columns else []
    st.write(f"- Кол-во фраз: **{n_phrases}**")
    st.write(f"- Уникальных тематик: **{len(all_topics)}**")
    st.write(f"- Средняя длина фразы: **{df['phrase'].astype(str).str.len().mean():.1f}**")
    st.write(f"- Максимальная длина фразы: **{df['phrase'].astype(str).str.len().max()}**")

    st.subheader("🧰 Логгирование (вкл/выкл)")
    show_debug = st.toggle("Показывать отладочную информацию", value=True,
                           help="Это легко убрать на проде — выключите и удалите блоки sidebar.write().")

    st.subheader("💻 Мониторинг ресурсов")
    vm = psutil.virtual_memory()
    cpu_pct = psutil.cpu_percent(interval=None)
    rss = get_process_mem()
    st.write(f"CPU: **{cpu_pct:.1f}%**")
    st.write(f"RAM (process): **{human_bytes(rss)}**")
    st.write(f"RAM (system used): **{human_bytes(vm.used)} / {human_bytes(vm.total)} ({vm.percent:.1f}%)**")

# ---- Главная: выбор моделей и префиксов
st.markdown("### 🧠 Выбор моделей для A/B")
colA, colB = st.columns(2)

with colA:
    st.markdown("**Модель A**")
    model_a = st.selectbox("Выбери/введи HF id модели A:", options=PRESET_MODELS, index=1, key="model_a", help="Можно дописать свой id вручную.")
    model_a = st.text_input("Или укажи другой HF id для модели A:", value=model_a, key="model_a_text")
with colB:
    st.markdown("**Модель B**")
    model_b = st.selectbox("Выбери/введи HF id модели B:", options=PRESET_MODELS, index=0, key="model_b", help="Можно дописать свой id вручную.")
    model_b = st.text_input("Или укажи другой HF id для модели B:", value=model_b, key="model_b_text")

st.markdown("---")

# Префиксы и режимы
st.markdown("### 🔖 Префиксы и режимы")
c1, c2, c3, c4 = st.columns([1,1,1,1])
with c1:
    add_prefix_a = st.toggle("A: add_prefix", value=True, key="add_prefix_a")
with c2:
    add_prefix_b = st.toggle("B: add_prefix", value=True, key="add_prefix_b")
with c3:
    hybrid_a = st.toggle("A: гибридный запрос", value=True, key="hybrid_a", help="Смешивает query с префиксом и без — как в твоём проекте.")
with c4:
    hybrid_b = st.toggle("B: гибридный запрос", value=True, key="hybrid_b")

# Кастомные префиксы (опционально)
with st.expander("🧩 Пользовательские префиксы (необязательно)"):
    st.caption("Если оставить пустыми — подставятся автопрефиксы по семейству модели (E5/BGE).")
    colp1, colp2 = st.columns(2)
    with colp1:
        custom_q_a = st.text_input("A: query_prefix", value="", placeholder="например, 'query: '")
        custom_p_a = st.text_input("A: passage_prefix", value="", placeholder="например, 'passage: '")
    with colp2:
        custom_q_b = st.text_input("B: query_prefix", value="", placeholder="например, 'query: '", key="q_b")
        custom_p_b = st.text_input("B: passage_prefix", value="", placeholder="например, 'document: '", key="p_b")

def none_if_empty(s: str) -> Optional[str]:
    s = (s or "").strip()
    return s if s != "" else None

custom_q_a = none_if_empty(custom_q_a)
custom_p_a = none_if_empty(custom_p_a)
custom_q_b = none_if_empty(custom_q_b)
custom_p_b = none_if_empty(custom_p_b)

# Предвычисление эмбеддингов под каждую модель
with st.spinner("Готовим эмбеддинги под модель A..."):
    precomputed_a = compute_phrase_embeddings(df, model_a, add_prefix_a, custom_q_a, custom_p_a)
with st.spinner("Готовим эмбеддинги под модель B..."):
    precomputed_b = compute_phrase_embeddings(df, model_b, add_prefix_b, custom_q_b, custom_p_b)

# Показ «лёгких» метрик по моделям
mc1, mc2 = st.columns(2)
with mc1:
    st.markdown(f"**Модель A:** `{model_a}`")
    st.write(f"- Размерность эмбеддингов: **{precomputed_a['dim']}**")
    st.write(f"- Фраз в базе: **{len(df)}**")
with mc2:
    st.markdown(f"**Модель B:** `{model_b}`")
    st.write(f"- Размерность эмбеддингов: **{precomputed_b['dim']}**")
    st.write(f"- Фраз в базе: **{len(df)}**")

st.markdown("---")

# ====== Поисковый ввод + фильтры ======
st.markdown("### 🔎 Поиск и A/B сравнение")
query = st.text_input("Введите запрос (query):", "")

top_k = st.slider("Top-K результатов:", min_value=1, max_value=20, value=DEFAULT_TOP_K, step=1)

# Фильтр по тематикам (опционально)
all_topics_sorted = sorted({t for row in df["topics"] for t in row}) if "topics" in df.columns else []
selected_topics = st.multiselect("Фильтр по тематикам:", all_topics_sorted, default=[])
if selected_topics:
    mask = df["topics"].apply(lambda ts: any(t in ts for t in selected_topics))
    df_filtered = df[mask].reset_index(drop=True)

    # Вырезаем соответствующие строки из эмбеддингов
    idxs = np.where(mask.values)[0]
    precomputed_a_f = {
        "embeddings": precomputed_a["embeddings"][idxs] if len(idxs) else np.zeros((0, precomputed_a["dim"]), dtype="float32"),
        "norms": precomputed_a["norms"][idxs] if len(idxs) else np.zeros((0,), dtype="float32"),
        "dim": precomputed_a["dim"]
    }
    precomputed_b_f = {
        "embeddings": precomputed_b["embeddings"][idxs] if len(idxs) else np.zeros((0, precomputed_b["dim"]), dtype="float32"),
        "norms": precomputed_b["norms"][idxs] if len(idxs) else np.zeros((0,), dtype="float32"),
        "dim": precomputed_b["dim"]
    }
else:
    df_filtered = df
    precomputed_a_f = precomputed_a
    precomputed_b_f = precomputed_b

# ====== A/B поиск ======
if query:
    # До/после ресурсы — лёгкое логгирование
    cpu_before = psutil.cpu_percent(interval=None)
    mem_before = get_process_mem()

    res_a = search_topk(
        query_text=query,
        df=df_filtered,
        model_name=model_a,
        precomputed=precomputed_a_f,
        add_prefix=add_prefix_a,
        custom_query_prefix=custom_q_a,
        custom_passage_prefix=custom_p_a,
        top_k=top_k,
        hybrid_query=hybrid_a
    )
    res_b = search_topk(
        query_text=query,
        df=df_filtered,
        model_name=model_b,
        precomputed=precomputed_b_f,
        add_prefix=add_prefix_b,
        custom_query_prefix=custom_q_b,
        custom_passage_prefix=custom_p_b,
        top_k=top_k,
        hybrid_query=hybrid_b
    )

    cpu_after = psutil.cpu_percent(interval=None)
    mem_after = get_process_mem()

    ca, cb = st.columns(2)
    with ca:
        st.subheader("A — результаты")
        st.write(f"⏱️ Время ответа: **{res_a['latency_s']*1000:.1f} ms**")
        st.write(f"CPU Δ: **{max(0.0, cpu_after - cpu_before):.1f}%**, RAM Δ: **{human_bytes(max(0, mem_after - mem_before))}**")
        for item in res_a["results"]:
            with st.container():
                st.markdown(
                    f"""<div style="border:1px solid #e0e0e0;border-radius:12px;padding:12px;margin:8px 0;background:#fafafa">
                        <div style="font-weight:600">🧠 {item['phrase_full']}</div>
                        <div style="font-size:13px;color:#666">🎯 Score: {item['score']:.3f}</div>
                        <div style="font-size:13px;color:#666">🔖 Тематики: {', '.join(item['topics']) if item['topics'] else '—'}</div>
                    </div>""",
                    unsafe_allow_html=True
                )

    with cb:
        st.subheader("B — результаты")
        st.write(f"⏱️ Время ответа: **{res_b['latency_s']*1000:.1f} ms**")
        st.write(f"CPU Δ: **{max(0.0, cpu_after - cpu_before):.1f}%**, RAM Δ: **{human_bytes(max(0, mem_after - mem_before))}**")
        for item in res_b["results"]:
            with st.container():
                st.markdown(
                    f"""<div style="border:1px solid #e0e0e0;border-radius:12px;padding:12px;margin:8px 0;background:#fafafa">
                        <div style="font-weight:600">🧠 {item['phrase_full']}</div>
                        <div style="font-size:13px;color:#666">🎯 Score: {item['score']:.3f}</div>
                        <div style="font-size:13px;color:#666">🔖 Тематики: {', '.join(item['topics']) if item['topics'] else '—'}</div>
                    </div>""",
                    unsafe_allow_html=True
                )

    if show_debug:
        st.sidebar.write("### 🧾 Отладка (по запросу)")
        st.sidebar.write(f"Модель A: `{model_a}` | add_prefix={add_prefix_a} | hybrid={hybrid_a}")
        st.sidebar.write(f"Модель B: `{model_b}` | add_prefix={add_prefix_b} | hybrid={hybrid_b}")
        st.sidebar.write(f"CPU (before→after): {cpu_before:.1f}% → {cpu_after:.1f}%")
        st.sidebar.write(f"RAM (proc, before→after): {human_bytes(mem_before)} → {human_bytes(mem_after)}")

st.markdown("---")

# ====== Симуляция нагрузки ======
st.markdown("### 🧪 Нагрузочный тест (простая симуляция параллельных пользователей)")
col_s1, col_s2, col_s3, col_s4 = st.columns([1,1,1,2])
with col_s1:
    sim_model = st.selectbox("Модель для симуляции", options=[model_a, model_b], index=0)
with col_s2:
    sim_users = st.number_input("Кол-во пользователей", min_value=1, max_value=32, value=5, step=1)
with col_s3:
    sim_reqs = st.number_input("Запросов на пользователя", min_value=1, max_value=50, value=3, step=1)
with col_s4:
    sim_query = st.text_input("Тестовый запрос для симуляции", value=query or "как оплатить заказ")

if st.button("🚀 Запустить симуляцию"):
    if sim_model == model_a:
        pre = precomputed_a_f
        ap = add_prefix_a
        cq = custom_q_a
        cp = custom_p_a
    else:
        pre = precomputed_b_f
        ap = add_prefix_b
        cq = custom_q_b
        cp = custom_p_b

    with st.spinner("Выполняем параллельные запросы..."):
        stats = simulate_users(
            n_users=int(sim_users),
            n_requests_per_user=int(sim_reqs),
            query_text=sim_query,
            df=df_filtered,
            model_name=sim_model,
            precomputed=pre,
            add_prefix=ap,
            custom_query_prefix=cq,
            custom_passage_prefix=cp,
            top_k=top_k,
        )
    st.success("Готово!")

    colr1, colr2, colr3 = st.columns(3)
    with colr1:
        st.metric("Запросов всего", stats["count"])
        st.metric("Среднее, ms", f"{stats['avg_ms']:.1f}")
        st.metric("p95, ms", f"{stats['p95_ms']:.1f}")
    with colr2:
        st.metric("min, ms", f"{stats['min_ms']:.1f}")
        st.metric("max, ms", f"{stats['max_ms']:.1f}")
        st.metric("Время теста, s", f"{stats['total_time_s']:.2f}")
    with colr3:
        st.metric("CPU start → end", f"{stats['cpu_start_pct']:.1f}% → {stats['cpu_end_pct']:.1f}%")
        st.metric("RAM start", human_bytes(stats["mem_start"]))
        st.metric("RAM end", human_bytes(stats["mem_end"]))

st.markdown("---")
st.caption("Подсказка: на проде можно убрать весь отладочный вывод (sidebar.write), оставив только нужные метрики.")
