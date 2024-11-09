# Retrieval-Augmented Generation Vehicle Troubleshooter

- [1. Introduction](#1-introduction)
- [2. Setup](#2-setup)
- [3. How to Use](#3-using-the-program)
- [4. Additional Configuration](#4-additional-configuration-optional)
- [5. Cleanup](#5-cleanup-optional)
- [6. Ideas for the Future](#6-ideas-for-the-future)

## 1. Introduction

This project was created for the Fall 2024 NVIDIA and LlamaIndex Developer Contest.

The purpose is to aggregate user-provided data and online sources to enable a chatbot to help users resolve issues with their vehicle.

The following tools were used:

1. [`Gradio`](https://www.gradio.app/) for the user interface

2. [`llama-3.1-8b-instruct`](https://build.nvidia.com/meta/llama-3.2-70b-instruct) LLM model for the chatbot

3. [`llama-3.1-11b-vision-instruct`](https://build.nvidia.com/meta/llama-3.2-11b-vision-instruct) for image analysis

4. [`embed-qa-4`](https://build.nvidia.com/nvidia/embed-qa-4) to create vector embeddings from text

5. [`NVIDIA Nemo Curator`](https://github.com/NVIDIA/NeMo-Curator) to curate vehicle data and troubleshooting information to the chatbot

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
conda env create --name rag-troubleshooter.venv
conda activate rag-troubleshooter.venv
pip install -r requirements.txt
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

## 4. Additional Configuration (Optional)

No curated data is included by default. 

If you would like to use `CommonCrawl` data, include the following command line arguments:

```bash
# Use CommonCrawl dataset when it already exists
python app.py --source commoncrawl

# Download and use CommonCrawl dataset with a specified URL limit to download
python app.py --source commoncrawl --download_url_limit 5
```

If you would like to use local data as your dataset, use these arguments:

```bash
# Use local dataset at default location 'sample_data'
python app.py --source local

# Use local dataset at user-preferred location (Replace <ENTER_DIRECTORY_HERE> with directory)
python app.py --source local --location <ENTER_DIRECTORY_HERE>
```

The sample documents in `sample_documents` may be manually uploaded in the file upload location when the program is running. Additionally, `sample_data` includes a car manual. These are included for testing/demonstration purposes. If you would like to tune the dataset for a different vehicle while using `local` data, remove/replace the included car manual with your own or specify a different location.

## 5. Cleanup (Optional) 

When you are done using the program, you can destroy the `Anaconda` virtual environment as follows:

```bash
conda deactivate
conda env remove --name rag-troubleshooter.venv
```

This will free space on your system occupied by the program dependencies.

## 6. Ideas for the Future

1. AR application to give step-by-step visual overlay instructions on each generated repair procedure.

2. Add vehicle troubleshooting forum data to curation dataset to help generate unconventional solutions.

3. Generate images and diagrams for particular vehicle to help convey technical instructions.