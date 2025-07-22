#!/usr/bin/env python3
"""
Test script for FitJourney MediaPipe Service
Run this script to verify the service is working correctly
"""

import requests
import json
import time
from typing import Dict, Any

# Configuration
SERVICE_URL = "http://localhost:8000"  # Change to your deployed URL
TEST_IMAGE_URL = "https://example.com/test-image.jpg"  # Replace with actual test image

def test_health_endpoint() -> bool:
    """Test the health check endpoint."""
    print("🔍 Testing health endpoint...")
    try:
        response = requests.get(f"{SERVICE_URL}/health", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Health check passed: {data}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check error: {str(e)}")
        return False

def test_root_endpoint() -> bool:
    """Test the root endpoint."""
    print("\n🔍 Testing root endpoint...")
    try:
        response = requests.get(f"{SERVICE_URL}/", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Root endpoint passed: {data}")
            return True
        else:
            print(f"❌ Root endpoint failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Root endpoint error: {str(e)}")
        return False

def test_body_analysis(image_url: str = None) -> Dict[str, Any]:
    """Test the body analysis endpoint."""
    print("\n🔍 Testing body analysis endpoint...")
    
    # Use a publicly available test image or the provided URL
    test_url = image_url or "https://images.unsplash.com/photo-1571019613454-1cb2f99b2d8b?w=400"
    
    payload = {
        "image_url": test_url,
        "user_height": 175.0,
        "user_weight": 70.0
    }
    
    try:
        print(f"📷 Analyzing image: {test_url}")
        response = requests.post(
            f"{SERVICE_URL}/analyze-body",
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Analysis completed successfully!")
            print(f"📊 Results:")
            print(f"   - Success: {data.get('success')}")
            print(f"   - Landmarks detected: {data.get('landmarks_detected')}")
            print(f"   - Visibility: {data.get('landmark_visibility')}")
            print(f"   - Waist-hip ratio: {data.get('waist_hip_ratio')}")
            print(f"   - Shoulder-waist ratio: {data.get('shoulder_waist_ratio')}")
            print(f"   - Body shape: {data.get('body_shape')}")
            print(f"   - Estimated body fat: {data.get('estimated_body_fat')}%")
            print(f"   - Posture score: {data.get('posture_score')}")
            return data
        else:
            print(f"❌ Analysis failed: {response.status_code}")
            print(f"Response: {response.text}")
            return {}
    except Exception as e:
        print(f"❌ Analysis error: {str(e)}")
        return {}

def test_performance() -> None:
    """Test service performance with multiple requests."""
    print("\n🚀 Testing performance...")
    
    test_url = "https://images.unsplash.com/photo-1571019613454-1cb2f99b2d8b?w=400"
    num_requests = 3
    total_time = 0
    
    for i in range(num_requests):
        print(f"  Request {i+1}/{num_requests}...")
        start_time = time.time()
        
        payload = {
            "image_url": test_url,
            "user_height": 175.0,
            "user_weight": 70.0
        }
        
        try:
            response = requests.post(
                f"{SERVICE_URL}/analyze-body",
                json=payload,
                timeout=30
            )
            end_time = time.time()
            request_time = end_time - start_time
            total_time += request_time
            
            if response.status_code == 200:
                print(f"    ✅ Request {i+1} completed in {request_time:.2f}s")
            else:
                print(f"    ❌ Request {i+1} failed: {response.status_code}")
        except Exception as e:
            print(f"    ❌ Request {i+1} error: {str(e)}")
    
    avg_time = total_time / num_requests
    print(f"📈 Performance summary:")
    print(f"   - Total requests: {num_requests}")
    print(f"   - Average response time: {avg_time:.2f}s")
    print(f"   - Total time: {total_time:.2f}s")

def test_error_handling() -> None:
    """Test error handling with invalid inputs."""
    print("\n🧪 Testing error handling...")
    
    # Test with invalid URL
    print("  Testing invalid URL...")
    payload = {
        "image_url": "https://invalid-url-that-does-not-exist.com/image.jpg",
        "user_height": 175.0,
        "user_weight": 70.0
    }
    
    try:
        response = requests.post(
            f"{SERVICE_URL}/analyze-body",
            json=payload,
            timeout=30
        )
        data = response.json()
        if not data.get('success'):
            print(f"    ✅ Properly handled invalid URL: {data.get('message')}")
        else:
            print(f"    ❌ Should have failed with invalid URL")
    except Exception as e:
        print(f"    ❌ Unexpected error: {str(e)}")
    
    # Test with non-image URL
    print("  Testing non-image URL...")
    payload = {
        "image_url": "https://www.google.com",
        "user_height": 175.0,
        "user_weight": 70.0
    }
    
    try:
        response = requests.post(
            f"{SERVICE_URL}/analyze-body",
            json=payload,
            timeout=30
        )
        data = response.json()
        if not data.get('success'):
            print(f"    ✅ Properly handled non-image URL: {data.get('message')}")
        else:
            print(f"    ❌ Should have failed with non-image URL")
    except Exception as e:
        print(f"    ❌ Unexpected error: {str(e)}")

def main():
    """Run all tests."""
    print("🚀 FitJourney MediaPipe Service Test Suite")
    print("=" * 50)
    
    # Test basic endpoints
    health_ok = test_health_endpoint()
    root_ok = test_root_endpoint()
    
    if not health_ok or not root_ok:
        print("\n❌ Basic endpoints failed. Please check service status.")
        return
    
    # Test main functionality
    analysis_result = test_body_analysis()
    
    if analysis_result.get('success'):
        print("\n✅ Core functionality working!")
        
        # Additional tests
        test_performance()
        test_error_handling()
        
        print("\n🎉 All tests completed!")
        print("\n📋 Test Summary:")
        print("   ✅ Health check: PASSED")
        print("   ✅ Root endpoint: PASSED")
        print("   ✅ Body analysis: PASSED")
        print("   ✅ Performance test: COMPLETED")
        print("   ✅ Error handling: TESTED")
        print("\n🚀 Service is ready for integration!")
        
    else:
        print("\n❌ Core functionality failed. Please check:")
        print("   - Service is running")
        print("   - Image URL is accessible")
        print("   - MediaPipe dependencies are installed")
        print("   - Check service logs for detailed errors")

if __name__ == "__main__":
    # Allow user to specify custom service URL
    import sys
    if len(sys.argv) > 1:
        SERVICE_URL = sys.argv[1]
        print(f"Using custom service URL: {SERVICE_URL}")
    
    main() 