import os
import subprocess
import sys

def create_venv_and_install_requirements():
    # Define the virtual environment directory
    venv_dir = "venv"

    # Step 1: Create the virtual environment
    print("Creating virtual environment...")
    subprocess.run([sys.executable, "-m", "venv", venv_dir], check=True)
    print(f"Virtual environment created in '{venv_dir}'.")

    # Step 2: Activate the virtual environment
    activate_script = os.path.join(venv_dir, "Scripts", "activate") if os.name == "nt" else os.path.join(venv_dir, "bin", "activate")
    print(f"To activate the virtual environment, run: source {activate_script}")

    # Step 3: Install dependencies from requirements.txt
    requirements_file = "requirements.txt"
    if os.path.exists(requirements_file):
        print(f"Installing dependencies from '{requirements_file}'...")
        subprocess.run([os.path.join(venv_dir, "bin", "pip") if os.name != "nt" else os.path.join(venv_dir, "Scripts", "pip"), "install", "-r", requirements_file], check=True)
        print("Dependencies installed successfully.")
    else:
        print(f"'{requirements_file}' not found. Please ensure it exists in the current directory.")

if __name__ == "__main__":
    try:
        create_venv_and_install_requirements()
    except subprocess.CalledProcessError as e:
        print(f"An error occurred: {e}")
