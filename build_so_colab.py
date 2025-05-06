#!/usr/bin/env python3
import os
import subprocess
from tree_sitter import Language

# Print tree-sitter version for debugging
try:
    import tree_sitter
    print(f"Tree-sitter version: {tree_sitter.__version__}")
except (ImportError, AttributeError):
    print("Could not determine tree-sitter version")

# Define paths
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
BUILD_DIR = os.path.join(REPO_ROOT, "build")
LANGUAGES_SO = os.path.join(BUILD_DIR, "my-languages.so")

# Create build directory if it doesn't exist
os.makedirs(BUILD_DIR, exist_ok=True)

# Define language repositories and their paths
# Adjust these paths based on your Colab setup
C_LANG_DIR = "/content/drive/MyDrive/Defect/tree-sitter-c"

# Verify the C language repository exists
if not os.path.exists(C_LANG_DIR):
    raise FileNotFoundError(f"C language repository not found at {C_LANG_DIR}")

# Check if we're on the right version
os.chdir(C_LANG_DIR)
result = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"], 
                        capture_output=True, text=True)
current_branch = result.stdout.strip()
print(f"Current tree-sitter-c branch/tag: {current_branch}")

# Build the language
try:
    Language.build_library(
        LANGUAGES_SO,
        [C_LANG_DIR]
    )
    print(f"Successfully built language library at {LANGUAGES_SO}")
    
    # Test loading the language
    lang = Language(LANGUAGES_SO, 'c')
    print("Successfully loaded C language")
    
    # Copy to multiple locations for easier access
    for path in ["/content/my-languages.so", "/content/drive/MyDrive/Defect/my-languages.so"]:
        dir_path = os.path.dirname(path)
        if os.path.exists(dir_path):
            subprocess.run(["cp", LANGUAGES_SO, path])
            print(f"Copied language library to {path}")
    
except Exception as e:
    print(f"Error building language library: {str(e)}")
