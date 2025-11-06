import streamlit as st
import requests
from PIL import Image
import io
import time
from typing import List
import config

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


def check_api_health() -> dict:
    """Check API health status."""
    try:
        response = requests.get(config.API_HEALTH_ENDPOINT, timeout=5)
        if response.status_code == 200:
            return {"status": "healthy", "data": response.json()}
        return {"status": "unhealthy", "error": f"Status code: {response.status_code}"}
    except Exception as e:
        return {"status": "unavailable", "error": str(e)}


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

        st.divider()

        st.header("ℹ️ About")
        st.info(
            """
            **Current Status: Phase 2 Complete**

            ✅ **Phase 1:** OCR text extraction
            ✅ **Phase 2:** Text parsing & dish extraction

            Current capabilities:
            - Upload 1-5 menu images
            - Extract text using Tesseract OCR
            - Parse dishes with names and prices
            - View confidence scores

            **Coming Soon:**
            - Phase 3: Vegetarian classification
            - Phase 4: MCP server integration
            - Phase 5: LLM classification
            - Phase 6: RAG for confidence scoring
            """
        )

        st.divider()

        st.header("📝 Instructions")
        st.markdown(
            """
            1. Upload menu images (JPEG, PNG, WEBP)
            2. Click "Extract Text" button
            3. Review OCR results
            4. Check confidence scores
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

        # Extract text button
        if st.button("🔍 Extract Text", type="primary", use_container_width=True):
            if health["status"] != "healthy":
                st.error("❌ API is not available. Please start the API server first.")
                return

            with st.spinner("🔄 Extracting text from images..."):
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
                    response = requests.post(
                        config.API_EXTRACT_TEXT_ENDPOINT,
                        files=files,
                        timeout=60,
                    )
                    elapsed_time = (time.time() - start_time) * 1000

                    if response.status_code == 200:
                        results = response.json()

                        st.success(f"✅ Text extraction completed in {elapsed_time:.2f}ms")

                        # Display results
                        st.header("📄 OCR Results")

                        for idx, result in enumerate(results):
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
