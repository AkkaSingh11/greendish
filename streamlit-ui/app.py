import streamlit as st
import requests
from PIL import Image
import io
import time
from typing import List
import ui_config as config

# Page configuration
st.set_page_config(
    page_title="ConvergeFi - Menu Analyzer",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .status-healthy {
        color: #28a745;
        font-weight: bold;
    }
    .status-unhealthy {
        color: #dc3545;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

MODE_OPTIONS = {
    "🥗 Non-AI (Keyword Pipeline)": "non-ai",
    "🤖 AI Agent (LangGraph)": "ai",
}
READY_MODE = "non-ai"


def check_api_health() -> dict:
    """Check API health status."""
    try:
        response = requests.get(config.API_HEALTH_ENDPOINT, timeout=5)
        if response.status_code == 200:
            return {"status": "healthy", "data": response.json()}
        return {"status": "unhealthy", "error": f"Status code: {response.status_code}"}
    except Exception as e:
        return {"status": "unavailable", "error": str(e)}


def check_mcp_health() -> dict:
    """Check MCP server health status."""
    try:
        response = requests.get(config.MCP_HEALTH_ENDPOINT, timeout=5)
        if response.status_code == 200:
            return {"status": "healthy", "data": response.json()}
        return {"status": "unhealthy", "error": f"Status code: {response.status_code}"}
    except Exception as e:
        return {"status": "unavailable", "error": str(e)}


def fetch_mcp_tools() -> dict:
    """Fetch registered MCP tools from the API service."""
    try:
        response = requests.get(config.API_MCP_TOOLS_ENDPOINT, timeout=5)
        if response.status_code == 200:
            return {"status": "ok", "tools": response.json()}
        return {
            "status": "error",
            "error": f"Status code: {response.status_code}",
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


def main():
    """Main Streamlit application."""

    # Header
    st.markdown('<div class="main-header">🍽️ ConvergeFi Menu Analyzer</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Restaurant Menu Vegetarian Dish Analyzer - Dashboard</div>',
        unsafe_allow_html=True,
    )

    # Sidebar
    with st.sidebar:
        st.header("📊 System Status")

        # API Health Check
        health = check_api_health()
        if health["status"] == "healthy":
            st.markdown('<p class="status-healthy">✅ API: Healthy</p>', unsafe_allow_html=True)
            health_data = health.get("data", {})
            st.caption(f"Version: {health_data.get('version', 'N/A')}")
        else:
            st.markdown('<p class="status-unhealthy">❌ API: Unavailable</p>', unsafe_allow_html=True)
            st.error(f"Error: {health.get('error', 'Unknown error')}")
            st.warning("Please start the API server:\n```bash\ncd api\npython main.py\n```")

        # MCP Server Health
        mcp_health = check_mcp_health()
        if mcp_health["status"] == "healthy":
            st.markdown('<p class="status-healthy">✅ MCP Server: Healthy</p>', unsafe_allow_html=True)
            mcp_data = mcp_health.get("data", {})
            st.caption(f"Service: {mcp_data.get('service', 'calculator')}")
        elif mcp_health["status"] == "unhealthy":
            st.markdown('<p class="status-unhealthy">❌ MCP Server: Unhealthy</p>', unsafe_allow_html=True)
            st.error(f"Error: {mcp_health.get('error', 'Unknown error')}")
        else:
            st.markdown('<p class="status-unhealthy">⚠️ MCP Server: Unavailable</p>', unsafe_allow_html=True)
            st.error(f"Error: {mcp_health.get('error', 'Unknown error')}")

        with st.expander("🔧 View MCP Tools"):
            if mcp_health["status"] != "healthy":
                st.info("Start the MCP server to inspect available tools.")
            else:
                tools_response = fetch_mcp_tools()
                if tools_response["status"] == "ok":
                    tools = tools_response.get("tools", [])
                    if not tools:
                        st.warning("No MCP tools are currently registered.")
                    else:
                        for tool in tools:
                            st.write(f"**{tool.get('title') or tool.get('name')}**")
                            st.caption(tool.get("description") or "No description provided.")
                            if tool.get("input_schema"):
                                st.caption("Input schema")
                                st.json(tool["input_schema"])
                            if tool.get("output_schema"):
                                st.caption("Output schema")
                                st.json(tool["output_schema"])
                            st.divider()
                else:
                    st.error(f"Unable to fetch MCP tools: {tools_response.get('error')}")

        st.divider()

        st.header("ℹ️ About")
        st.info(
            """
            **Current Progress: Phase 8 Complete**

            ✅ **Phase 1:** OCR text extraction  
            ✅ **Phase 2:** Text parsing & dish extraction  
            ✅ **Phase 3:** Keyword-based vegetarian classification  
            ✅ **Phase 4:** MCP calculator integration  
            ✅ **Phase 5:** Non-AI pipeline exposed end-to-end  
            ✅ **Phase 6:** OpenRouter client ready (API pending wiring)  
            ✅ **Phase 7:** LangGraph agent scaffolded  
            ✅ **Phase 8:** RAG store seeded & Streamlit explorer added  

            **Available Today:**
            - Upload 1-5 menu images (non-AI pipeline)
            - Inspect RAG evidence via *Phase 8 — RAG Explorer* page

            **Coming Soon (AI pipeline):**
            - Phase 9: Agent integration into `/process-menu?mode=ai`
            - Phase 10: LangSmith tracing & analytics
            """
        )

        st.divider()

        st.header("📝 Instructions")
        st.markdown(
            """
            1. Upload menu images (JPEG, PNG, WEBP)  
            2. Click "Process Menu"  
            3. Review OCR text, parsed dishes, and vegetarian classifications
            """
        )

    # Main content
    st.header("📤 Upload Menu Images")

    # File uploader
    uploaded_files = st.file_uploader(
        "Choose menu images",
        type=config.SUPPORTED_FORMATS,
        accept_multiple_files=True,
        help=f"Upload up to {config.MAX_IMAGES} menu images",
    )

    if uploaded_files:
        if len(uploaded_files) > config.MAX_IMAGES:
            st.error(f"❌ Too many files! Maximum {config.MAX_IMAGES} images allowed.")
            return

        st.success(f"✅ {len(uploaded_files)} image(s) uploaded successfully")

        # Display uploaded images
        st.subheader("📸 Uploaded Images")
        cols = st.columns(min(len(uploaded_files), 3))
        for idx, uploaded_file in enumerate(uploaded_files):
            with cols[idx % 3]:
                image = Image.open(uploaded_file)
                st.image(image, caption=uploaded_file.name, use_container_width=True)
                st.caption(f"Size: {image.size[0]}x{image.size[1]}")

        st.divider()

        # Process menu button
        st.subheader("⚙️ Processing Mode")
        mode_label = st.radio(
            "Select processing pipeline",
            list(MODE_OPTIONS.keys()),
            index=0,
            help="Use the LangGraph agent for AI-assisted classification (requires OPENROUTER_API_KEY).",
        )
        selected_mode = MODE_OPTIONS[mode_label]
        ai_mode_selected = selected_mode == "ai"

        use_rag = False
        if ai_mode_selected:
            st.info(
                "🤖 AI agent mode will call OpenRouter via LangGraph. Make sure the API service has "
                "OPENROUTER_API_KEY configured and the MCP server is running."
            )
            use_rag = st.toggle(
                "Use RAG fallback for low-confidence dishes",
                value=False,
                help="Enable Retrieval-Augmented Generation re-checks when the initial classification confidence is low.",
            )

        if st.button(
            "🧠 Process Menu",
            type="primary",
            use_container_width=True,
        ):
            if health["status"] != "healthy":
                st.error("❌ API is not available. Please start the API server first.")
                return

            with st.spinner("🔄 Processing menu images..."):
                try:
                    # Prepare files for upload
                    files = []
                    for uploaded_file in uploaded_files:
                        uploaded_file.seek(0)  # Reset file pointer
                        files.append(
                            ("files", (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type))
                        )

                    # Make API request
                    start_time = time.time()
                    params = {"mode": selected_mode}
                    if ai_mode_selected:
                        params["use_rag"] = use_rag
                    response = requests.post(
                        config.API_PROCESS_MENU_ENDPOINT,
                        files=files,
                        params=params,
                        timeout=60,
                    )
                    elapsed_time = (time.time() - start_time) * 1000

                    if response.status_code == 200:
                        results = response.json()

                        st.success(f"✅ Menu processed in {elapsed_time:.2f}ms")
                        result_mode = results.get("mode", selected_mode)
                        st.caption(f"Processing mode: {result_mode}")

                        # Display results
                        st.header("📄 OCR Results")

                        ocr_results: List[dict] = results.get("ocr_results", [])

                        for idx, result in enumerate(ocr_results):
                            with st.expander(
                                f"📝 {result['image_name']} - {result['confidence']:.1f}% confidence",
                                expanded=True,
                            ):
                                col1, col2 = st.columns([3, 1])

                                with col1:
                                    st.subheader("Extracted Text")
                                    if result["raw_text"]:
                                        st.text_area(
                                            "Text",
                                            result["raw_text"],
                                            height=300,
                                            key=f"text_{idx}",
                                            label_visibility="collapsed",
                                        )
                                    else:
                                        st.warning("⚠️ No text detected in this image")

                                with col2:
                                    st.subheader("Metrics")
                                    st.metric("Confidence", f"{result['confidence']:.1f}%")
                                    st.metric("Processing Time", f"{result['processing_time_ms']:.2f}ms")
                                    st.metric("Text Length", f"{len(result['raw_text'])} chars")

                                    # Confidence indicator
                                    conf = result["confidence"]
                                    if conf >= 80:
                                        st.success("High Confidence")
                                    elif conf >= 60:
                                        st.warning("Medium Confidence")
                                    else:
                                        st.error("Low Confidence")

                        # Menu parsing summary
                        parsed_menu = results.get("parsed_menu", {})
                        dishes = results.get("dishes", [])
                        vegetarian_dishes = results.get("vegetarian_dishes", [])
                        calculation_summary = results.get("calculation_summary")

                        total_dishes = parsed_menu.get("total_dishes", len(dishes))
                        veg_count = len(vegetarian_dishes)
                        veg_percent = (veg_count / total_dishes * 100) if total_dishes else 0.0

                        st.header("🥗 Vegetarian Classification Summary")
                        col_summary = st.columns(4 if calculation_summary else 3)
                        col_summary[0].metric("Total Dishes", total_dishes)
                        col_summary[1].metric("Vegetarian Dishes", veg_count)
                        col_summary[2].metric("Vegetarian %", f"{veg_percent:.1f}%")

                        summary_reason = None
                        if calculation_summary:
                            col_summary[3].metric(
                                "Vegetarian Total",
                                f"${calculation_summary.get('total_price', 0.0):.2f}",
                            )
                            priced = calculation_summary.get("priced_dish_count")
                            missing = calculation_summary.get("missing_price_count")
                            parts = [
                                f"MCP Avg Confidence: {calculation_summary.get('average_confidence', 0):.2f}",
                            ]
                            if priced is not None:
                                parts.append(f"Priced dishes: {priced}")
                            if missing:
                                parts.append(f"Missing price: {missing}")
                            st.caption(" | ".join(parts))
                            summary_reason = calculation_summary.get("reasoning")

                        if vegetarian_dishes:
                            msg = (
                                f"{veg_count} of {total_dishes} dishes classified as vegetarian "
                                f"({veg_percent:.1f}%)."
                            )
                            if calculation_summary:
                                msg += f" MCP total: ${calculation_summary.get('total_price', 0.0):.2f}."
                            st.success(msg)
                            if summary_reason:
                                st.caption(summary_reason)
                        else:
                            st.warning("No vegetarian dishes detected.")

                        if dishes:
                            st.subheader("🍽️ Parsed Dishes")
                            for dish in dishes:
                                with st.expander(
                                    f"{'🥗' if dish.get('is_vegetarian') else '🍖'} {dish.get('name', 'Unknown')} "
                                    f"- Confidence {dish.get('confidence', 0) * 100:.0f}%"
                                ):
                                    st.write(f"**Price:** {dish.get('price', 'N/A')}")
                                    st.write(f"**Vegetarian:** {dish.get('is_vegetarian')}")
                                    st.write(f"**Method:** {dish.get('classification_method', 'n/a')}")
                                    st.write(f"**Reasoning:** {dish.get('reasoning', 'n/a')}")
                                    if dish.get("signals"):
                                        st.json(dish["signals"])
                                    st.code(dish.get("raw_text", ""), language="text")

                    else:
                        st.error(f"❌ API Error: {response.status_code}")
                        st.json(response.json())

                except requests.exceptions.RequestException as e:
                    st.error(f"❌ Network error: {str(e)}")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

    else:
        st.info("👆 Please upload menu images to get started")

        # Show example
        st.divider()
        st.subheader("📋 Example")
        st.markdown(
            """
            Sample menu images are available in the project root directory:
            - `menu1.jpeg`
            - `menu2.png`
            - `menu3.webp`

            Upload these files to test the OCR functionality!
            """
        )


if __name__ == "__main__":
    main()
