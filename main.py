import subprocess
import os
import sys

def run_servers():
    # Start backend server
    print("Starting backend server...")
    backend_process = subprocess.Popen([
        sys.executable, "-m", "uvicorn", 
        "app.api_gateway:app", 
        "--host", "0.0.0.0", 
        "--port", "8000", 
        "--reload"
    ])
    
    # Start frontend server
    print("Starting frontend server...")
    os.chdir("frontend/capt")
    frontend_process = subprocess.Popen(["npm", "start"])
    
    try:
        # Wait for both processes
        frontend_process.wait()
        backend_process.wait()
    except KeyboardInterrupt:
        print("\nShutting down servers...")
        frontend_process.terminate()
        backend_process.terminate()
        frontend_process.wait()
        backend_process.wait()
        print("Servers shut down successfully")

if __name__ == "__main__":
    run_servers()