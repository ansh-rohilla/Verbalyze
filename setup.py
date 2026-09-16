from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="verbalyze",
    version="1.0.0",
    author="Ansh Rohilla",
    description="Indic Voice AI Suite for Full-Duplex Indian Telephony, Speech Benchmarks & SLM Training",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ansh-rohilla/Verbalyze",
    packages=find_packages(include=["verbalyze", "verbalyze.*"]),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Multimedia :: Sound/Audio :: Speech",
        "Topic :: Communications :: Telephony",
    ],
    python_requires=">=3.9",
    install_requires=[
        "edge-tts>=6.1.10",
        "gTTS>=2.5.1",
        "sounddevice>=0.5.0",
        "SpeechRecognition>=3.14.0",
        "pydub>=0.25.1",
        "fastapi>=0.110.0",
        "uvicorn>=0.28.0",
        "httpx>=0.27.0",
        "requests>=2.31.0",
        "datasets>=2.18.0",
        "huggingface-hub>=0.22.0",
        "pandas>=2.0.0",
        "gradio>=6.0.0",
    ],
    entry_points={
        "console_scripts": [
            "verbalyze = verbalyze.cli:main",
        ],
    },
)
