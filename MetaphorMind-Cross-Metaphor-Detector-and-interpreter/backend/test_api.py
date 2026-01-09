"""
Simple API testing script for the Metaphor Detection backend
Run this after starting the backend server to verify all endpoints work
"""

import requests
import json
from typing import Dict, Any

BASE_URL = "http://localhost:8000"

def print_section(title: str):
    """Print a formatted section header"""
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60)

def print_response(response: requests.Response):
    """Print formatted response"""
    print(f"Status Code: {response.status_code}")
    try:
        data = response.json()
        print(f"Response: {json.dumps(data, indent=2, ensure_ascii=False)}")
    except:
        print(f"Response: {response.text}")

def test_health_check():
    """Test the health check endpoint"""
    print_section("Testing Health Check Endpoint")
    try:
        response = requests.get(f"{BASE_URL}/health")
        print_response(response)
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_prediction(text: str, expected_language: str):
    """Test the prediction endpoint"""
    print_section(f"Testing Prediction: {text[:50]}...")
    try:
        payload = {"text": text}
        response = requests.post(
            f"{BASE_URL}/predict",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        print_response(response)
        
        if response.status_code == 200:
            data = response.json()
            detected_lang = data.get("language")
            label = data.get("label")
            confidence = data.get("confidence")
            
            print(f"\n✅ Detected Language: {detected_lang}")
            print(f"✅ Label: {label}")
            print(f"✅ Confidence: {confidence:.2%}")
            
            if detected_lang == expected_language:
                print(f"✅ Language detection correct!")
                return True
            else:
                print(f"⚠️  Expected {expected_language}, got {detected_lang}")
                return False
        else:
            print(f"❌ Request failed")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_translation(text: str, language: str):
    """Test the translation endpoint"""
    print_section(f"Testing Translation: {text[:50]}...")
    try:
        payload = {
            "text": text,
            "source_language": language
        }
        response = requests.post(
            f"{BASE_URL}/translate",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        print_response(response)
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_empty_input():
    """Test error handling with empty input"""
    print_section("Testing Empty Input (Error Handling)")
    try:
        payload = {"text": ""}
        response = requests.post(
            f"{BASE_URL}/predict",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        print_response(response)
        
        if response.status_code == 400:
            print("✅ Correctly rejected empty input")
            return True
        else:
            print("⚠️  Should have returned 400 for empty input")
            return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_unsupported_language():
    """Test error handling with unsupported language"""
    print_section("Testing Unsupported Language (Error Handling)")
    try:
        payload = {"text": "This is English text"}
        response = requests.post(
            f"{BASE_URL}/predict",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        print_response(response)
        
        if response.status_code == 400:
            print("✅ Correctly rejected unsupported language")
            return True
        else:
            print("⚠️  Should have returned 400 for unsupported language")
            return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def run_all_tests():
    """Run all API tests"""
    print("\n" + "🚀"*30)
    print("  METAPHOR DETECTION API TEST SUITE")
    print("🚀"*30)
    
    results = []
    
    # Test 1: Health Check
    results.append(("Health Check", test_health_check()))
    
    # Test 2: Hindi Metaphor
    results.append((
        "Hindi Metaphor",
        test_prediction("वह आसमान छू रहा है", "hindi")
    ))
    
    # Test 3: Hindi Normal
    results.append((
        "Hindi Normal",
        test_prediction("मैं स्कूल जा रहा हूं", "hindi")
    ))
    
    # Test 4: Tamil Metaphor
    results.append((
        "Tamil Metaphor",
        test_prediction("அவன் வானத்தை தொடுகிறான்", "tamil")
    ))
    
    # Test 5: Tamil Normal
    results.append((
        "Tamil Normal",
        test_prediction("நான் பள்ளிக்கு செல்கிறேன்", "tamil")
    ))
    
    # Test 6: Kannada Metaphor
    results.append((
        "Kannada Metaphor",
        test_prediction("ಅವನು ಬೆಂಕಿಯಂತೆ ಕೋಪಗೊಂಡನು", "kannada")
    ))
    
    # Test 7: Kannada Normal
    results.append((
        "Kannada Normal",
        test_prediction("ನಾನು ಶಾಲೆಗೆ ಹೋಗುತ್ತಿದ್ದೇನೆ", "kannada")
    ))
    
    # Test 8: Translation
    results.append((
        "Translation",
        test_translation("वह आसमान छू रहा है", "hindi")
    ))
    
    # Test 9: Empty Input
    results.append((
        "Empty Input Error",
        test_empty_input()
    ))
    
    # Test 10: Unsupported Language
    results.append((
        "Unsupported Language Error",
        test_unsupported_language()
    ))
    
    # Print Summary
    print_section("TEST SUMMARY")
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\n{'='*60}")
    print(f"  Results: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    print(f"{'='*60}\n")
    
    if passed == total:
        print("🎉 All tests passed! Your API is working correctly.")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")

if __name__ == "__main__":
    print("\n⚠️  Make sure the backend server is running on http://localhost:8000")
    print("   Start it with: uvicorn main:app --reload\n")
    
    input("Press Enter to start testing...")
    
    try:
        run_all_tests()
    except KeyboardInterrupt:
        print("\n\n⚠️  Testing interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
