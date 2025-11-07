import requests
import streamlit as st

import ui_config as config

st.set_page_config(page_title="Phase 8 — RAG Explorer", page_icon="🧠", layout="centered")

st.title("🧠 Phase 8 — RAG Evidence Explorer")
st.write(
    """
Use the retrieval-augmented generation (RAG) store to inspect evidence that supports vegetarian
classification decisions. The dataset is chunked from `api/data/vegetarian_db.json` and embedded
into ChromaDB so you can immediately validate results without running the full LangGraph agent.
"""
)


@st.cache_data(show_spinner=False)
def has_rag_service() -> bool:
    try:
        response = requests.get(
            config.API_RAG_SEARCH_ENDPOINT,
            params={"query": "Margherita Pizza", "top_k": 1},
            timeout=5,
        )
        return response.status_code != 503
    except Exception:
        return False


service_enabled = has_rag_service()

if not service_enabled:
    st.error(
        "The API RAG service is disabled or unavailable. Make sure the FastAPI backend is running "
        "and that `settings.rag_enabled = True` (the default) in `api/config.py` before retrying."
    )
    st.stop()


with st.sidebar:
    st.header("⚙️ Settings")
    top_k = st.slider("Results (Top K)", min_value=1, max_value=10, value=3, step=1)
    st.caption("The API enforces a maximum of 10 matches per query.")

    st.header("🧹 Maintenance")
    st.write("Rebuild the vector store from the vegetarian dataset.")
    if st.button("Reseed RAG Data", type="primary"):
        try:
            response = requests.post(config.API_RAG_RESEED_ENDPOINT, json={"force": True}, timeout=20)
            if response.status_code == 202:
                data = response.json()
                st.success(f"Vector store reseeded with {data.get('documents', 0)} documents.")
                has_rag_service.clear()
            else:
                st.error(f"Failed to reseed: {response.status_code} — {response.text}")
        except Exception as exc:
            st.error(f"Reseed request failed: {exc}")

st.subheader("🔍 Query the Vector Store")

query = st.text_input("Dish name or snippet", placeholder="e.g. Garden Veggie Soup")

if st.button("Search", type="primary", disabled=not bool(query.strip())):
    try:
        params = {"query": query.strip(), "top_k": top_k}
        response = requests.get(config.API_RAG_SEARCH_ENDPOINT, params=params, timeout=10)
        if response.status_code == 200:
            data = response.json()
            matches = data.get("matches", [])
            if not matches:
                st.warning("No matches found. Try a different dish name or description.")
            else:
                st.success(f"Retrieved {len(matches)} match(es).")
                for idx, match in enumerate(matches, start=1):
                    category = match.get("category", "unknown").title()
                    score = match.get("score", 0.0)
                    st.markdown(f"**{idx}. {match.get('name', 'Unknown')}** — {category} (confidence {score:.2f})")
                    st.write(match.get("description", "No description available."))
                    st.caption(f"Chunk index: {match.get('chunk_index', 0)}")
                    st.divider()
        elif response.status_code == 503:
            st.error("RAG service unavailable. Verify the API logs and configuration.")
        else:
            st.error(f"Search failed: {response.status_code} — {response.text}")
    except Exception as exc:
        st.error(f"Request failed: {exc}")

st.info(
    "This page talks directly to the `/api/v1/rag` endpoints, making it easy to validate embedding "
    "quality and retrieval relevance before wiring the LangGraph agent into the API pipeline."
)
