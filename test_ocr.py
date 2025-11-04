#!/usr/bin/env python3
"""Quick test script for OCR API."""

import requests
import json
import sys

API_URL = "http://localhost:8000"

def test_health():
    """Test health endpoint."""
    print("=" * 60)
    print("Testing health endpoint...")
    print("=" * 60)

    response = requests.get(f"{API_URL}/health")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()
    return response.status_code == 200

def test_ocr(image_path):
    """Test OCR extraction."""
    print("=" * 60)
    print(f"Testing OCR with: {image_path}")
    print("=" * 60)

    try:
        with open(image_path, 'rb') as f:
            files = {'files': (image_path, f, 'image/jpeg')}
            response = requests.post(f"{API_URL}/api/v1/extract-text", files=files)

        print(f"Status Code: {response.status_code}")

        if response.status_code == 200:
            results = response.json()
            for result in results:
                print(f"\nImage: {result['image_name']}")
                print(f"Confidence: {result['confidence']:.1f}%")
                print(f"Processing Time: {result['processing_time_ms']:.2f}ms")
                print(f"Text Length: {len(result['raw_text'])} characters")
                print("\nExtracted Text (first 500 chars):")
                print("-" * 60)
                print(result['raw_text'][:500])
                print("-" * 60)
            return True
        else:
            print(f"Error: {response.text}")
            return False

    except Exception as e:
        print(f"Error: {str(e)}")
        return False

def main():
    """Run all tests."""
    if not test_health():
        print("❌ Health check failed!")
        sys.exit(1)

    print("\n✅ API is healthy!\n")

    # Test with menu images
    images = ["menu1.jpeg", "menu2.png", "menu3.webp"]

    for image in images:
        success = test_ocr(image)
        if success:
            print(f"\n✅ {image} processed successfully!\n")
        else:
            print(f"\n❌ {image} failed!\n")

if __name__ == "__main__":
    main()
