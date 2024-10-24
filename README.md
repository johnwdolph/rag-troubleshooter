# Retrieval-Augmented Generation Vehicle Troubleshooter

- [1. Introduction](#1-introduction)
- [2. Setup](#2-setup)
- [3. How to Use](#3-using-the-program)
- [4. Cleanup](#4-cleanup-optional)

## 1. Introduction

This project was created for the Fall 2024 NVIDIA and LlamaIndex Developer Contest.

The purpose is to aggregate user-provided data and online sources to enable a chatbot to help users resolve issues with their vehicle.

The following tools were used:

1. [`Gradio`](https://www.gradio.app/) for the user interface

2. [`llama-3.1-70b-instruct`](https://build.nvidia.com/meta/llama-3.2-70b-instruct) LLM model for the chatbot

3. [`llama-3.1-11b-vision-instruct`](https://build.nvidia.com/meta/llama-3.2-11b-vision-instruct) for image analysis

4. [`NVIDIA Nemo Curator`](https://github.com/NVIDIA/NeMo-Curator) to curate vehicle data and troubleshooting information to the chatbot

5. [`NVIDIA Guardrails`](https://github.com/NVIDIA/NeMo-Guardrails) to make sure the chatbot stays on the topic of vehicles

## 2. Setup

1. Install the Python dependencies:

Install `Nemo` dependencies:

**Note**: These commands assume you are using `bash` or another similar shell program.

```bash
sudo apt update && sudo apt install -y libsndfile1 ffmpeg
pip install Cython
```
2. (**Optional**) Setup `Anaconda` virtual environment:

**Note**: You are not required to create a virtual environment, but will need to install dependencies manually otherwise.

```bash
conda env create --file environment.yml --name rag-troubleshooter.venv
conda activate rag-troubleshooter.venv
```

3. Obtain and NVIDIA API key:

Visit [build.nvidia.com](https://build.nvidia.com/meta/llama-3_1-405b-instruct) and click `Get API Key`. Paste this key into the `.env` file at location `NVIDIA_API_KEY`.

## 3. Using the Program

1. Run the program:

```bash
python app.py
```

2. Fill out the information at the top of the program interface.

2. Once you submit, the chatbot will automatically respond using your provided information as context. You may ask it additional questions thereafter.

## 4. Cleanup (Optional) 

When you are done using the program, you can destroy the `Anaconda` virtual environment as follows:

```bash
conda deactivate
conda env remove --name rag-troubleshooter.venv
```

This will free space on your system occupied by the program dependencies.