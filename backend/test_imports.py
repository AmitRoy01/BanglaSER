"""Test if all imports work"""
print("Testing imports...")

try:
    import os
    print("✓ os")
except Exception as e:
    print(f"✗ os: {e}")

try:
    import torch
    print(f"✓ torch {torch.__version__}")
except Exception as e:
    print(f"✗ torch: {e}")

try:
    import torchaudio
    print(f"✓ torchaudio")
except Exception as e:
    print(f"✗ torchaudio: {e}")

try:
    import soundfile as sf
    print("✓ soundfile")
except Exception as e:
    print(f"✗ soundfile: {e}")

try:
    import librosa
    print("✓ librosa")
except Exception as e:
    print(f"✗ librosa: {e}")

try:
    from fastapi import FastAPI
    print("✓ fastapi")
except Exception as e:
    print(f"✗ fastapi: {e}")

try:
    import uvicorn
    print("✓ uvicorn")
except Exception as e:
    print(f"✗ uvicorn: {e}")

print("\nChecking model file...")
model_path = "../bangla_ser_best.pth"
if os.path.exists(model_path):
    print(f"✓ Model file exists at: {os.path.abspath(model_path)}")
else:
    print(f"✗ Model file NOT found at: {os.path.abspath(model_path)}")
    print(f"Current directory: {os.getcwd()}")

print("\nAll checks complete!")
