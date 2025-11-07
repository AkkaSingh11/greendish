import streamlit as st
import requests
from PIL import Image
import time
from typing import List, Dict
import pandas as pd
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import ui_config as config

# Page configuration
st.set_page_config(
    page_title="Phase 2: Parser Test - ConvergeFi",
    page_icon="📋",
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
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .status-healthy {
        color: #28a745;
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


def process_menu_images(uploaded_files: List) -> Dict:
    """Process menu images through the API."""
    files = []
    for uploaded_file in uploaded_files:
        uploaded_file.seek(0)  # Reset file pointer
        files.append(
            ("files", (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type))
        )

    # Make API request to process-menu endpoint
    start_time = time.time()
    response = requests.post(
        config.API_PROCESS_MENU_ENDPOINT,
        files=files,
        timeout=60,
    )
    elapsed_time = (time.time() - start_time) * 1000

    if response.status_code == 200:
        result = response.json()
        result["client_elapsed_ms"] = elapsed_time
        return result
    else:
        raise Exception(f"API Error {response.status_code}: {response.text}")


def display_parsing_stats(parsed_menu: Dict, dishes: List[Dict]):
    """Display parsing statistics."""
    if not parsed_menu and not dishes:
        st.warning("No dishes parsed")
        return

    total_dishes = parsed_menu.get("total_dishes") if parsed_menu else len(dishes)
    dishes_with_prices = parsed_menu.get("dishes_with_prices") if parsed_menu else sum(
        1 for d in dishes if d.get("price") is not None
    )
    dishes_without_prices = parsed_menu.get("dishes_without_prices") if parsed_menu else max(
        total_dishes - dishes_with_prices, 0
    )

    avg_confidence = parsed_menu.get("average_confidence") if parsed_menu else (
        sum(d.get("confidence", 0.0) for d in dishes) / total_dishes if total_dishes else 0
    )
    price_coverage = parsed_menu.get("price_coverage") if parsed_menu else (
        dishes_with_prices / total_dishes if total_dishes > 0 else 0
    )
    avg_ocr_confidence = parsed_menu.get("average_ocr_confidence") if parsed_menu else None

    # Display metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Dishes", total_dishes)

    with col2:
        st.metric("With Prices", dishes_with_prices,
                  delta=f"{price_coverage:.0%} coverage")

    with col3:
        st.metric("Without Prices", dishes_without_prices)

    with col4:
        st.metric("Avg Confidence", f"{avg_confidence:.2f}")

    if avg_ocr_confidence is not None:
        st.caption(f"Average OCR Confidence: {avg_ocr_confidence:.2f}%")


def main():
    """Main Streamlit application for Phase 2."""

    # Header
    st.markdown('<div class="main-header">📋 Phase 2: Parser Test</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Test text parsing and dish structuring from OCR results</div>',
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

        st.divider()

        st.header("ℹ️ Phase 2 Features")
        st.info(
            """
            **Parser Capabilities:**

            ✅ Extract dish names from OCR text
            ✅ Parse prices ($X.XX, X.XX formats)
            ✅ Handle multi-line dishes
            ✅ Handle dishes without prices
            ✅ Calculate confidence scores
            ✅ Parsing statistics

            **What's Next:**
            - Phase 3: Keyword classification
            - Phase 4: MCP server integration
            - Phase 5: LLM classification
            """
        )

        st.divider()

        st.header("📝 Test Instructions")
        st.markdown(
            """
            1. Upload menu images
            2. Click "Process Menu"
            3. Review parsed dishes table
            4. Check parsing statistics
            5. Inspect individual dishes
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

        # Display uploaded images preview
        with st.expander("📸 Preview Uploaded Images", expanded=False):
            cols = st.columns(min(len(uploaded_files), 3))
            for idx, uploaded_file in enumerate(uploaded_files):
                with cols[idx % 3]:
                    image = Image.open(uploaded_file)
                    st.image(image, caption=uploaded_file.name, use_container_width=True)
                    st.caption(f"Size: {image.size[0]}x{image.size[1]}")

        st.divider()

        # Process menu button
        if st.button("🔍 Process Menu", type="primary", use_container_width=True):
            if health["status"] != "healthy":
                st.error("❌ API is not available. Please start the API server first.")
                return

            with st.spinner("🔄 Processing menu images..."):
                try:
                    # Process menu
                    result = process_menu_images(uploaded_files)

                    st.success(
                        f"✅ Processing completed in {result['processing_time_ms']:.2f}ms "
                        f"(total with network: {result['client_elapsed_ms']:.2f}ms)"
                    )

                    # Display OCR Summary
                    st.header("📄 OCR Summary")
                    ocr_cols = st.columns(len(result['ocr_results']))
                    for idx, ocr_result in enumerate(result['ocr_results']):
                        with ocr_cols[idx]:
                            st.metric(
                                ocr_result['image_name'],
                                f"{ocr_result['confidence']:.1f}%",
                                delta=f"{len(ocr_result['raw_text'])} chars"
                            )

                    st.divider()

                    # Display Parsing Statistics
                    st.header("📊 Parsing Statistics")
                    parsed_menu = result.get('parsed_menu', {})
                    dishes = result.get('dishes', [])
                    display_parsing_stats(parsed_menu, dishes)

                    st.divider()

                    if parsed_menu:
                        st.header("🧾 Structured Menu JSON")
                        st.json(parsed_menu)

                        st.divider()

                    # Display Parsed Dishes Table
                    st.header("🍽️ Parsed Dishes")

                    if dishes:
                        # Convert to DataFrame for better display
                        df_data = []
                        for idx, dish in enumerate(dishes, 1):
                            df_data.append({
                                "#": idx,
                                "Dish Name": dish.get("name", "N/A"),
                                "Price": f"${dish['price']:.2f}" if dish.get("price") is not None else "N/A",
                                "Confidence": f"{dish.get('confidence', 0):.2f}",
                                "Raw Text": dish.get("raw_text", "")[:50] + "..."
                            })

                        df = pd.DataFrame(df_data)

                        # Display table
                        st.dataframe(
                            df,
                            use_container_width=True,
                            hide_index=True,
                            column_config={
                                "#": st.column_config.NumberColumn(width="small"),
                                "Dish Name": st.column_config.TextColumn(width="large"),
                                "Price": st.column_config.TextColumn(width="small"),
                                "Confidence": st.column_config.TextColumn(width="small"),
                                "Raw Text": st.column_config.TextColumn(width="medium"),
                            }
                        )

                        # Detailed view
                        st.subheader("🔍 Detailed Dish Information")

                        # Filter options
                        col1, col2 = st.columns(2)
                        with col1:
                            show_with_prices = st.checkbox("Show only dishes with prices", value=False)
                        with col2:
                            show_without_prices = st.checkbox("Show only dishes without prices", value=False)

                        filtered_dishes = dishes
                        if show_with_prices:
                            filtered_dishes = [d for d in dishes if d.get("price") is not None]
                        elif show_without_prices:
                            filtered_dishes = [d for d in dishes if d.get("price") is None]

                        # Display individual dishes
                        for idx, dish in enumerate(filtered_dishes, 1):
                            price_str = f"${dish['price']:.2f}" if dish.get("price") is not None else "No price"
                            conf_str = f"{dish.get('confidence', 0):.2f}"

                            with st.expander(
                                f"#{idx} {dish['name']} - {price_str} (confidence: {conf_str})",
                                expanded=False
                            ):
                                col1, col2 = st.columns([2, 1])

                                with col1:
                                    st.write("**Raw Text:**")
                                    st.code(dish.get("raw_text", "N/A"))

                                with col2:
                                    st.write("**Details:**")
                                    st.write(f"**Name:** {dish['name']}")
                                    if dish.get("price") is not None:
                                        st.write(f"**Price:** ${dish['price']:.2f}")
                                    else:
                                        st.write("**Price:** Not found")
                                    st.write(f"**Confidence:** {dish.get('confidence', 0):.2f}")

                                    # Confidence indicator
                                    conf = dish.get("confidence", 0)
                                    if conf >= 0.8:
                                        st.success("High Confidence")
                                    elif conf >= 0.6:
                                        st.warning("Medium Confidence")
                                    else:
                                        st.error("Low Confidence")

                    else:
                        st.warning("⚠️ No dishes were parsed from the menu images")

                    # Debug info
                    with st.expander("🐛 Debug Info", expanded=False):
                        st.json(result)

                except requests.exceptions.RequestException as e:
                    st.error(f"❌ Network error: {str(e)}")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())

    else:
        st.info("👆 Please upload menu images to get started")

        # Show example
        st.divider()
        st.subheader("📋 Sample Menus")
        st.markdown(
            """
            Test the parser with these sample menu images from the project root:
            - **menu1.jpeg** - Applebee's menu (complex, multi-column layout)
            - **menu2.png** - Simple menu with clear prices
            - **menu3.webp** - Cafe menu with descriptions

            Expected results:
            - Menu 1: 20-30 dishes parsed
            - Menu 2: 5-10 dishes parsed
            - Menu 3: 15-25 dishes parsed
            """
        )


if __name__ == "__main__":
    main()
