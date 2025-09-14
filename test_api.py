#!/usr/bin/env python3
"""
Simple test script to verify the Soccer Analytics API is working
"""

import requests
import json
import sys
from pathlib import Path

def test_api():
    """Test the API endpoints"""
    base_url = "http://localhost:5000/api"
    
    print("🧪 Testing Soccer Analytics API...")
    print("=" * 40)
    
    try:
        # Test health endpoint
        print("1. Testing health endpoint...")
        response = requests.get(f"{base_url}/health", timeout=5)
        if response.status_code == 200:
            print("   ✅ Health check passed")
            print(f"   Response: {response.json()}")
        else:
            print(f"   ❌ Health check failed: {response.status_code}")
            return False
            
        # Test status endpoint
        print("\n2. Testing status endpoint...")
        response = requests.get(f"{base_url}/status", timeout=5)
        if response.status_code == 200:
            print("   ✅ Status check passed")
            print(f"   Response: {response.json()}")
        else:
            print(f"   ❌ Status check failed: {response.status_code}")
            return False
            
        print("\n✅ All API tests passed!")
        print("\n🚀 Ready for frontend integration!")
        print("\nNext steps:")
        print("1. Start your React dev server: npm run dev")
        print("2. Navigate to /analyze page")
        print("3. Upload a video and try live analysis")
        
        return True
        
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to API server")
        print("\nPlease start the API server first:")
        print("python start_api.py")
        return False
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        return False

if __name__ == "__main__":
    success = test_api()
    sys.exit(0 if success else 1)
