import os
import time
import re
import math
import psutil
import io
from io import BytesIO, StringIO
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
import streamlit as st
import torch
from sentence_transformers import SentenceTransformer

# =========================================
# ============== НАСТРОЙКИ ================
# =========================================

st.set_page_config(page_title="A/B-тестер эмбеддинговых моделей", layout="wide")

DEFAULT_DATASET_PATH = "https://raw.githubusercontent.com/skatzrskx55q/data-assistant-vfiziki/main/data6.xlsx"
DEFAULT_TOP_K = 5

# Правила префиксов по семействам
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
]

# =========================================
# =========== КЕШИ / ВСПОМОГАТЕЛЬНЫЕ ======
# =========================================

def detect_family_prefixes(model_name: str) -> Tuple[Optional[str], Optional[str]]:
    for rule in FAMILY_PREFIX_RULES:
        if re.search(rule["pattern"], model_name or ""):
            return rule["query_prefix"], rule["passage_prefix"]
    return None, None

def maybe_prefix(texts: List[str], prefix: Optional[str], add_prefix: bool) -> List[str]:
    if add_prefix and prefix:
        return [prefix + t for t in texts]
    return texts

def cosine_sim_matrix(vec: np.ndarray, mat: np.ndarray, mat_norms: np.ndarray) -> np.ndarray:
    vnorm = np.linalg.norm(vec) or 1e-10
    return (mat @ vec) / (mat_norms * vnorm)

def human_bytes(n_bytes: int) -> str:
    if n_bytes < 1024: return f"{n_bytes} B"
    for unit in ["KB","MB","GB","TB","PB"]:
        n_bytes /= 1024.0
        if n_bytes < 1024.0:
            return f"{n_bytes:.1f} {unit}"
    return f"{n_bytes:.1f} EB"

def get_process_mem():
    return psutil.Process(os.getpid()).memory_info().rss

@st.cache_data(show_spinner=False)
def load_excel_any(path_or_bytes: bytes | str) -> pd.DataFrame:
    if isinstance(path_or_bytes, (bytes, bytearray)):
        df = pd.read_excel(BytesIO(path_or_bytes))
    else:
        df = pd.read_excel(path_or_bytes)

    if "phrase" not in df.columns:
        raise ValueError("В Excel не найдена колонка 'phrase'")
    if "comment" not in df.columns:
        df["comment"] = ""

    # нормализация
    df["phrase_full"] = df["phrase"]
    df["phrase_proc"] = (
        df["phrase"].astype(str).str.lower().str.replace(r"\s+", " ", regex=True).str.strip()
    )
    # topics больше не используем — оставим пустые колонки для совместимости
    if "topics" not in df.columns:
        df["topics"] = [[] for _ in range(len(df))]
    return df[["phrase", "phrase_proc", "phrase_full", "topics", "comment"]]

# =========================================
# ====== ПОДГОТОВКА ЭМБЕДДИНГОВ БАЗЫ ======
# =========================================

@st.cache_resource(show_spinner=False)
def load_model(model_name: str) -> SentenceTransformer:
    # Загружаем модель на CPU (без автокастов), чтобы беречь память
    return SentenceTransformer(model_name, device="cpu")

@st.cache_data(show_spinner=False)
def compute_phrase_embeddings(
    df: pd.DataFrame,
    model_name: str,
    add_prefix: bool,
    custom_query_prefix: str | None,
    custom_passage_prefix: str | None,
    batch_size: int = 128,
) -> Dict[str, np.ndarray]:
    model = load_model(model_name)
    auto_q, auto_p = detect_family_prefixes(model_name)
    p_prefix = (custom_passage_prefix if (custom_passage_prefix is not None) else auto_p)
    passages = maybe_prefix(df["phrase_proc"].tolist(), p_prefix, add_prefix)

    embs: List[np.ndarray] = []
    for i in range(0, len(passages), batch_size):
        batch = passages[i:i+batch_size]
        batch_embs = model.encode(batch, convert_to_numpy=True, show_progress_bar=False, batch_size=batch_size)
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
# ========= ПОИСК / СИМУЛЯЦИЯ =============
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
    model = load_model(model_name)
    auto_q, _ = detect_family_prefixes(model_name)
    q_prefix = (custom_query_prefix if (custom_query_prefix is not None) else auto_q)

    phrase_embs = precomputed["embeddings"]
    phrase_norms = precomputed["norms"]
    if phrase_embs is None or phrase_embs.size == 0:
        return {"results": [], "latency_s": 0.0}

    t0 = time.time()
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

    from concurrent.futures import ThreadPoolExecutor, as_completed
    with ThreadPoolExecutor(max_workers=n_users) as ex:
        futures = [ex.submit(one_call) for _ in range(n_users * n_requests_per_user)]
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
        "latencies_ms": latencies * 1000.0,
    }

# =========================================
# =============== UI ======================
# =========================================

st.title("🔬 Универсальный A/B-тестер моделей эмбеддингов")
st.caption("Экономичный режим для Streamlit Community — ручная загрузка моделей, A/B-переключатель и отчёты.")

# ------ Сайдбар: Датасет / Ресурсы / Служебные
with st.sidebar:
    st.header("⚙️ Настройки")

    st.subheader("📥 Датасет")
    uploaded = st.file_uploader("Загрузить Excel (.xlsx) с колонкой 'phrase'", type=["xlsx"])
    if uploaded is not None:
        df = load_excel_any(uploaded.read())
        st.success(f"Загружено строк: {len(df)}")
    else:
        df = load_excel_any(DEFAULT_DATASET_PATH)
        st.info(f"Используем {DEFAULT_DATASET_PATH}. Строк: {len(df)}")

    st.subheader("🧪 Лёгкие метрики датасета")
    st.write(f"- Кол-во фраз: **{len(df)}**")
    st.write(f"- Средняя длина фразы: **{df['phrase'].astype(str).str.len().mean():.1f}**")
    st.write(f"- Максимальная длина фразы: **{df['phrase'].astype(str).str.len().max()}**")

    st.subheader("💻 Мониторинг ресурсов")
    vm = psutil.virtual_memory()
    cpu_pct = psutil.cpu_percent(interval=None)
    rss = get_process_mem()
    st.write(f"CPU: **{cpu_pct:.1f}%**")
    st.write(f"RAM (process): **{human_bytes(rss)}**")
    st.write(f"RAM (system used): **{human_bytes(vm.used)} / {human_bytes(vm.total)} ({vm.percent:.1f}%)**")

    st.subheader("🧹 Обслуживание")
    if st.button("♻️ Полный сброс (кэш + сессия)"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.session_state.clear()
        st.rerun()

# --------- Режим работы: A/B или одиночный
ab_mode = st.toggle("🔀 A/B тест включён", value=True, help="Выключите, чтобы тестировать одну модель.")

st.markdown("### 🧠 Модели")
# Инициализация структуры с загруженными моделями и индексами
if "models" not in st.session_state:
    st.session_state.models = {}     # name -> {loaded:bool, add_prefix, hybrid, custom_q, custom_p, batch_size, precomputed}
if "order" not in st.session_state:
    st.session_state.order = []      # порядок добавления

def model_controls(slot_name: str):
    with st.container(border=True):
        st.markdown(f"**{slot_name}**")
        col = st.columns([3,1,1,1])
        with col[0]:
            m_name = st.text_input(f"{slot_name}: HF id модели", value="", placeholder="например, intfloat/multilingual-e5-small", key=f"{slot_name}_name")
        with col[1]:
            add_prefix = st.toggle("add_prefix", value=True, key=f"{slot_name}_addpref")
        with col[2]:
            hybrid = st.toggle("гибрид", value=True, key=f"{slot_name}_hybrid")
        with col[3]:
            batch_size = st.number_input("batch", min_value=8, max_value=512, step=8, value=128, key=f"{slot_name}_bs")

        cq, cp = st.columns(2)
        with cq:
            custom_q = st.text_input("query_prefix (необязательно)", value="", key=f"{slot_name}_q")
        with cp:
            custom_p = st.text_input("passage_prefix (необязательно)", value="", key=f"{slot_name}_p")

        lc = st.columns([1,1,2])
        with lc[0]:
            if st.button("📦 Загрузить", key=f"{slot_name}_load"):
                name = (m_name or "").strip()
                if name == "":
                    st.warning("Укажите HF id модели.")
                else:
                    # реальная загрузка модели
                    with st.spinner("Загрузка модели..."):
                        _ = load_model(name)  # кэшируется как ресурс; создаёт экземпляр и скачивает при первом вызове
                    # предвычисление индекса
                    with st.spinner("Подготовка эмбеддингов базы..."):
                        pre = compute_phrase_embeddings(
                            df, name, add_prefix,
                            (custom_q.strip() or None),
                            (custom_p.strip() or None),
                            batch_size=batch_size
                        )
                    st.session_state.models[name] = dict(
                        add_prefix=add_prefix, hybrid=hybrid,
                        custom_q=(custom_q.strip() or None),
                        custom_p=(custom_p.strip() or None),
                        batch_size=int(batch_size),
                        precomputed=pre,
                    )
                    if name not in st.session_state.order:
                        st.session_state.order.append(name)
                    st.success(f"Модель '{name}' готова.")
        with lc[1]:
            if st.button("🗑️ Удалить", key=f"{slot_name}_del"):
                name = st.session_state.get(f"{slot_name}_name","")
                if name in st.session_state.models:
                    del st.session_state.models[name]
                    st.session_state.order = [x for x in st.session_state.order if x != name]
                    st.info(f"Модель '{name}' удалена из сессии. (Для освобождения RAM кэшов нажмите «Полный сброс» в сайдбаре)")

if ab_mode:
    cA, cB = st.columns(2)
    with cA: model_controls("Модель A")
    with cB: model_controls("Модель B")
else:
    model_controls("Модель")

st.markdown("---")

# --------- Таблица параметров загруженных моделей
def model_meta_table(names: List[str]) -> pd.DataFrame:
    rows = []
    for name in names:
        try:
            m = load_model(name)
            # попытка извлечь сведения из Transformer
            try:
                mod0 = m[0].auto_model  # первый модуль — Transformer
                cfg = getattr(mod0, "config", None)
            except Exception:
                mod0, cfg = None, None

            params = sum(p.numel() for p in m._first_module().auto_model.parameters()) if hasattr(m._first_module(), "auto_model") else \
                     sum(p.numel() for p in m._first_module().parameters())
            bytes_per_param = 4  # float32 оценка
            size_mb = params * bytes_per_param / (1024**2)

            dim = m.get_sentence_embedding_dimension()
            max_len = m.get_max_seq_length() if hasattr(m, "get_max_seq_length") else getattr(cfg, "max_position_embeddings", None)
            n_layers = getattr(cfg, "num_hidden_layers", None)
            hidden = getattr(cfg, "hidden_size", None)
            heads = getattr(cfg, "num_attention_heads", None)
            dtype = str(next(m.parameters()).dtype).replace("torch.", "") if hasattr(m, "parameters") else "unknown"
            device = str(next(iter(m._first_module().parameters())).device) if hasattr(m._first_module(), "parameters") else "cpu"

            row = dict(
                model=name,
                emb_dim=dim,
                max_seq_len=max_len,
                device=device,
                dtype=dtype,
                params=int(params),
                size_est_mb=round(size_mb, 1),
                layers=n_layers,
                hidden_size=hidden,
                attn_heads=heads,
                batch_opt=st.session_state.models[name]["batch_size"] if name in st.session_state.models else None,
                add_prefix=st.session_state.models.get(name,{}).get("add_prefix"),
                hybrid=st.session_state.models.get(name,{}).get("hybrid"),
                query_prefix=st.session_state.models.get(name,{}).get("custom_q"),
                passage_prefix=st.session_state.models.get(name,{}).get("custom_p"),
            )
            rows.append(row)
        except Exception as e:
            rows.append({"model": name, "error": str(e)})
    return pd.DataFrame(rows)

loaded_names = st.session_state.order[:]
if loaded_names:
    st.markdown("### 📋 Параметры загруженных моделей")
    meta_df = model_meta_table(loaded_names)
    st.dataframe(meta_df, use_container_width=True)
else:
    st.info("Пока моделей не загружено. Введите HF id и нажмите «Загрузить».")

st.markdown("---")

# --------- Поиск (одна или две модели, только среди загруженных)
st.markdown("### 🔎 Поиск")
query = st.text_input("Введите запрос:", "")
top_k = st.slider("Top-K результатов:", min_value=1, max_value=20, value=DEFAULT_TOP_K, step=1)

def run_search_block(model_name: str, title: str):
    if model_name not in st.session_state.models:
        st.warning(f"{title}: модель не загружена.")
        return None
    cfg = st.session_state.models[model_name]
    cpu_before = psutil.cpu_percent(interval=None)
    mem_before = get_process_mem()

    res = search_topk(
        query_text=query,
        df=df,
        model_name=model_name,
        precomputed=cfg["precomputed"],
        add_prefix=cfg["add_prefix"],
        custom_query_prefix=cfg["custom_q"],
        custom_passage_prefix=cfg["custom_p"],
        top_k=top_k,
        hybrid_query=cfg["hybrid"],
    )

    cpu_after = psutil.cpu_percent(interval=None)
    mem_after = get_process_mem()

    st.subheader(title)
    st.write(f"⏱️ Время ответа: **{res['latency_s']*1000:.1f} ms**")
    st.write(f"CPU Δ: **{max(0.0, cpu_after - cpu_before):.1f}%**, RAM Δ: **{human_bytes(max(0, mem_after - mem_before))}**")
    for item in res["results"]:
        with st.container():
            st.markdown(
                f"""<div style="border:1px solid #e0e0e0;border-radius:12px;padding:12px;margin:8px 0;background:#fafafa">
                    <div style="font-weight:600">🧠 {item['phrase_full']}</div>
                    <div style="font-size:13px;color:#666">🎯 Score: {item['score']:.3f}</div>
                </div>""",
                unsafe_allow_html=True
            )
    return res

if query:
    if ab_mode:
        c1, c2 = st.columns(2)
        with c1:
            # берём первую загруженную как A (если поле совпадает по имени — отлично)
            nameA = st.session_state.get("Модель A_name","")
            nameA = nameA if nameA in loaded_names else (loaded_names[0] if loaded_names else "")
            res_a = run_search_block(nameA, "A — результаты") if nameA else None
        with c2:
            nameB = st.session_state.get("Модель B_name","")
            nameB = nameB if (nameB in loaded_names and nameB != nameA) else (loaded_names[1] if len(loaded_names)>1 else "")
            res_b = run_search_block(nameB, "B — результаты") if nameB else None

        # скачать результаты A/B (если обе есть)
        if (res_a or res_b):
            rows = []
            if res_a:
                for r in res_a["results"]:
                    rows.append({"model":"A", "model_name": nameA, "query": query, "phrase": r["phrase_full"], "score": r["score"], "comment": r["comment"]})
            if res_b:
                for r in res_b["results"]:
                    rows.append({"model":"B", "model_name": nameB, "query": query, "phrase": r["phrase_full"], "score": r["score"], "comment": r["comment"]})
            results_df = pd.DataFrame(rows)
            csv_buf = io.StringIO()
            results_df.to_csv(csv_buf, index=False)
            st.download_button(
                label="📥 Скачать A/B результаты (CSV)",
                data=csv_buf.getvalue(),
                file_name="ab_results.csv",
                mime="text/csv",
            )
    else:
        # одиночный режим
        name = st.session_state.get("Модель_name","")
        name = name if name in loaded_names else (loaded_names[0] if loaded_names else "")
        res_single = run_search_block(name, "Результаты") if name else None
        if res_single:
            rows = [{"model": name, "query": query, "phrase": r["phrase_full"], "score": r["score"], "comment": r["comment"]} for r in res_single["results"]]
            results_df = pd.DataFrame(rows)
            csv_buf = io.StringIO()
            results_df.to_csv(csv_buf, index=False)
            st.download_button(
                label="📥 Скачать результаты (CSV)",
                data=csv_buf.getvalue(),
                file_name="results.csv",
                mime="text/csv",
            )

st.markdown("---")

# --------- Нагрузочный тест (только по загруженным моделям)
st.markdown("### 🧪 Нагрузочный тест")
if not loaded_names:
    st.info("Сначала загрузите хотя бы одну модель.")
else:
    col_s1, col_s2, col_s3, col_s4 = st.columns([1,1,1,2])
    with col_s1:
        sim_model = st.selectbox("Модель для симуляции", options=loaded_names, index=0)
    with col_s2:
        sim_users = st.number_input("Кол-во пользователей", min_value=1, max_value=32, value=5, step=1)
    with col_s3:
        sim_reqs = st.number_input("Запросов на пользователя", min_value=1, max_value=50, value=3, step=1)
    with col_s4:
        sim_query = st.text_input("Тестовый запрос для симуляции", value="как оплатить заказ")

    if st.button("🚀 Запустить симуляцию"):
        cfg = st.session_state.models[sim_model]
        with st.spinner("Выполняем параллельные запросы..."):
            stats = simulate_users(
                n_users=int(sim_users),
                n_requests_per_user=int(sim_reqs),
                query_text=sim_query,
                df=df,
                model_name=sim_model,
                precomputed=cfg["precomputed"],
                add_prefix=cfg["add_prefix"],
                custom_query_prefix=cfg["custom_q"],
                custom_passage_prefix=cfg["custom_p"],
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

        # -------- CSV-отчёт со всеми результатами
        per_req = pd.DataFrame({
            "request_id": np.arange(1, stats["count"]+1, dtype=int),
            "model": sim_model,
            "users": int(sim_users),
            "reqs_per_user": int(sim_reqs),
            "query": sim_query,
            "top_k": int(top_k),
            "latency_ms": stats["latencies_ms"],
        })
        summary = pd.DataFrame([{
            "model": sim_model,
            "users": int(sim_users),
            "reqs_per_user": int(sim_reqs),
            "query": sim_query,
            "top_k": int(top_k),
            "count": stats["count"],
            "avg_ms": stats["avg_ms"],
            "p95_ms": stats["p95_ms"],
            "min_ms": stats["min_ms"],
            "max_ms": stats["max_ms"],
            "total_time_s": stats["total_time_s"],
            "cpu_start_pct": stats["cpu_start_pct"],
            "cpu_end_pct": stats["cpu_end_pct"],
            "mem_start": stats["mem_start"],
            "mem_end": stats["mem_end"],
        }])

        # Склеиваем в один CSV с маркером
        per_req["kind"] = "per_request"
        summary["kind"] = "summary"
        report_df = pd.concat([per_req, summary], ignore_index=True)
        buf = io.StringIO()
        report_df.to_csv(buf, index=False)
        st.download_button(
            label="📥 Скачать отчёт нагрузочного теста (CSV)",
            data=buf.getvalue(),
            file_name="load_test_report.csv",
            mime="text/csv",
        )

st.markdown("---")
st.caption("Подсказки: • Модели загружаются вручную, что экономит RAM. • Для полного освобождения памяти после экспериментов используйте «Полный сброс» в сайдбаре.")
