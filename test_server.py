import requests
import sys
import time

def test_connection(port=8080):
    """Test if the server is running on the specified port"""
    try:
        response = requests.get(f"http://localhost:{port}/")
        if response.status_code == 200:
            print(f"✅ Server is running on port {port}")
            print(f"✅ Response: {response.text[:100]}...")
            return True
        else:
            print(f"❌ Server returned status code {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print(f"❌ Could not connect to server on port {port}")
        print("   Make sure the server is running and the port is correct")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    # Try different ports
    ports = [8080, 5000, 3000, 8000]
    
    for port in ports:
        print(f"\nTesting connection on port {port}...")
        if test_connection(port):
            print(f"\n✅ Server is accessible at http://localhost:{port}")
            sys.exit(0)
    
    print("\n❌ Could not connect to the server on any port")
    print("   Please make sure the server is running")
    print("   Try running 'python app.py' in a terminal")
    sys.exit(1) 