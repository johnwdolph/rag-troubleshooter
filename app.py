import gradio as gr
import requests, base64
import time
import os
import mimetypes
from dotenv import load_dotenv

from llama_index.core import SimpleDirectoryReader, Settings, VectorStoreIndex, StorageContext, load_index_from_storage, Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.embeddings.nvidia import NVIDIAEmbedding
from llama_index.llms.nvidia import NVIDIA
from llama_index.postprocessor.nvidia_rerank import NVIDIARerank

import nemo_curator_utils as ncu

load_dotenv()

stream=True

#configure application
Settings.text_spllitter = SentenceSplitter(chunk_size=500)
Settings.embed_model = NVIDIAEmbedding(mode="NV-Embed-QA", truncate="END")
Settings.llm = NVIDIA(model="meta/llama-3.1-70b-instruct")

# Check for API key (should be in .env)
NVIDIA_API_KEY = ""
if os.getenv('NVIDIA_API_KEY') is None:
    raise ValueError("No NVIDIA_API_KEY set...")
else:
    NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")

index = None
query_engine = None

#create milvus vector store and index
vector_store = MilvusVectorStore(uri='./milvus_store_context.db', dim=1024, overwrite=True)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

def llama_vision_processor(image):
    global nvidia_api_key

    invoke_url = 'https://ai.api.nvidia.com/v1/gr/meta/llama-3.2-90b-vision-instruct/chat/completions'
    stream = True

    with open(image, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode()

    assert len(image_b64) < 180_000, \
    "To upload larger images, use the assets API (see docs)"

    headers = {
    f"Authorization": "Bearer {nvidia_api_key}",
    "Accept": "text/event-stream" if stream else "application/json"
    }

    payload = {
    "model": 'meta/llama-3.2-90b-vision-instruct',
    "messages": [
        {
        "role": "user",
        "content": f'What is in this image? <img src="data:image/png;base64,{image_b64}" />'
        }
    ],
    "max_tokens": 512,
    "temperature": 1.00,
    "top_p": 1.00,
    "stream": stream
    }

    response = requests.post(invoke_url, headers=headers, json=payload)
    print(response)

    result = ""
    if stream:
        for line in response.iter_lines():
            if line:
                result += line.decode("utf-8")

    return f"Image Analysis: {result}"

def process_issue_context(files, issue_description="", car_type="", OBD_codes=""):
    global index, query_engine, vector_store, storage_context
    try:
        documents = []
        if files:
            file_paths = [file.name for file in files]
            
            for file_path in file_paths:
                
                mime_type, _ = mimetypes.guess_type(file_path)
                if mime_type and mime_type.startswith('image/'):
                    print("image found")
                    image_analysis_result = llama_vision_processor(file_path)
                    documents.append(Document(text=image_analysis_result, doc_id=file_path))
                else:
                    directory = os.path.dirname(file_path)
                    reader = SimpleDirectoryReader(input_files=[file_path])
                    documents.extend(reader.load_data())

        issue_context = f"Car Type: {car_type}\nCar OBD P-Codes: {OBD_codes}\nIssue Description: {issue_description}"
        issue_doc = Document(text=issue_context, doc_id="issue-context")
        documents.append(issue_doc)

        index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)

        query_engine = index.as_query_engine(similarity_top_k=20, streaming=True)

        return f"Successfully loaded {len(documents)} documents from {len(file_paths)} files - "
    
    except Exception as e:
        return f"Error loading info: {str(e)} - "

def stream_response(message, history):
    global query_engine
    if query_engine is None:
        yield history + [("Please provide information first", None)]
        return
    try:
        response = query_engine.query(message)
        partial_repsonse = ""
        for text in response.response_gen:
            partial_repsonse += text
            yield history + [(message, partial_repsonse)]
    except Exception as e:
        yield history + [(message, f"Error processing query: {str(e)}")]

def get_diagnostic(chatbot):
    message = "What is the issue given my provided information?"
    stream_response(message, chatbot)
    return chatbot

system_message = "You are an assistant built to help diagnose issues with vehicles."
initial_messages = [gr.ChatMessage(role="system", content=system_message)]

# create gradio interface
with gr.Blocks() as app:
    gr.Markdown("# RAG Vehicle Troubleshooting Helper")

    with gr.Row():

        file_input = gr.File(label="Select files to load", file_count="multiple")

        # Create a text input for car type
        car_type = gr.Textbox(label="Car Type", placeholder="Enter the car make and model (e.g., Toyota Corolla 2015)")

        # Create a text input for OBD Codes
        OBD_codes = gr.Textbox(label="OBD Codes", placeholder="Enter OBD codes (e.g., P0171, P0420)")

        # Create a text area for the user to describe the issue
        issue_description = gr.Textbox(label="Issue Description", placeholder="Describe the problem you're experiencing with the car", lines=5)

    #Buttons
    generate_images = gr.Checkbox("Generate output images")
    submit_button = gr.Button("Submit")
    result_button = gr.Button("Get Diagnostic")

    user_params_output = gr.Textbox(label="User Input Status")

    chatbot = gr.Chatbot(value=[[None, "Hi, ask me anything!"]])
    msg = gr.Textbox(label="Enter your question", interactive=True)

    submit_button.click(process_issue_context, inputs=[file_input, chatbot, issue_description, car_type, OBD_codes], outputs=user_params_output)
    result_button.click(get_diagnostic, inputs=[file_input], outputs=[chatbot])

    clear_btn = gr.Button("Clear")

    #events
    msg.submit(stream_response, inputs=[msg, chatbot], outputs=[chatbot])
    msg.submit(lambda: "", outputs=[msg])
    clear_btn.click(lambda: None, None, chatbot, queue=False)

#launch interface
if __name__ == "__main__":
    app.launch()
