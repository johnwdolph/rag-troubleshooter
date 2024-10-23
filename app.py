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

load_dotenv()

send_system_messages = True

CHATBOT_LLM = 'meta/llama-3.1-70b-instruct'
CHATBOT_LLM2 = 'meta/llama-3.1-8b-instruct'
VISION_LLM = 'meta/llama-3.2-11b-vision-instruct'
VISION_LLM_2 = 'microsoft/phi-3.5-vision-instruct'

#configure application
Settings.text_spllitter = SentenceSplitter(chunk_size=500)
Settings.embed_model = NVIDIAEmbedding(mode="NV-Embed-QA", truncate="END")
Settings.llm = NVIDIA(model=CHATBOT_LLM2)

# Check for API key (should be in .env)
NVIDIA_API_KEY = ""
LLAMA_VISION_API_KEY = os.getenv("LLAMA_VISION_API_KEY")

if os.getenv('NVIDIA_API_KEY') is None:
    raise ValueError("No NVIDIA_API_KEY set...")
else:
    NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")

index = None
query_engine = None

#create milvus vector store and index
vector_store = MilvusVectorStore(uri='./milvus_store_context.db', dim=1024, overwrite=True)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# system messages
system_messages = ["You are an assistant built to help diagnose issues with vehicles.",
                   "When the user asks for help, try to reference the user-provided car type, codes, and description in context with your response.",
                   "Make sure to verify if the car type and codes provided exist before discussing them. If necessary, correct these for the user and remeber your corrections.",
                   "Try not to give generic advice unless the information is vague. Use the specific car type and p codes to figure out your solution."
                   ]

def llama_vision_processor(image):

    invoke_url = "https://ai.api.nvidia.com/v1/gr/meta/llama-3.2-11b-vision-instruct/chat/completions"
    stream = False

    with open(image, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode()

    assert len(image_b64) < 180_000, \
    "To upload larger images, use the assets API (see docs)"
    

    headers = {
    "Authorization": f"Bearer {NVIDIA_API_KEY}",
    "Accept": "text/event-stream" if stream else "application/json"
    }

    payload = {
    "model": 'meta/llama-3.2-11b-vision-instruct',
    "messages": [
        {
            "role": "system",
            "content": "Only provide information relevant to cars that may be useful in diagnosing issues"
        },
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

    # Extract the content from the response
    if response.status_code == 200:
        response_json = response.json()
        completion_text = response_json['choices'][0]['message']['content']
        return completion_text
    else:
        print(f"Error: {response.status_code}")
    
    return None

def process_issue_context(files, issue_description="", car_type="", OBD_codes=""):
    global index, query_engine, vector_store, storage_context
    try:
        documents = []
        file_paths = []
        if files:
            
            if files:
                file_paths = [file.name for file in files]
            
            for file_path in file_paths:
                
                mime_type, _ = mimetypes.guess_type(file_path)
                if mime_type and mime_type.startswith('image/'):
                    image_analysis_result = llama_vision_processor(file_path)
                    if image_analysis_result:
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
    generate_images = gr.Checkbox(label="Generate output images")
    submit_button = gr.Button("Submit")
    user_params_output = gr.Textbox(label="User Input Status")

    chatbot = gr.Chatbot(value=[gr.ChatMessage(role='assistant', content='Hi, ask me anything!')], type='messages')

    def user(user_message, history: list):
        return history + [{'role': 'user', 'content': user_message}]

    def bot(user_message, history: list):
        global query_engine, system_messages, send_system_messages

        if query_engine is None:
            history.append({'role': 'assistant', 'content': "Hi, please provide some information first."})
            yield history
            return

        if send_system_messages:
            for msg in system_messages:
                history.append({'role': 'system', 'content': msg})
            send_system_messages = False  # Disable further system messages

        bot_message = query_engine.query(user_message)

        history.append({'role': 'assistant', 'content': ""})
        for character in bot_message.response_gen:
            history[-1]['content'] += character
            time.sleep(0.05)
            yield history

    msg = gr.Textbox(label="Enter your question", interactive=True)

    auto_query = gr.State("What is wrong and how can I solve it? Please give me a concise step-by-step solution if possible. Add proper line breaks between steps for readability.")
    submit_button.click(process_issue_context, inputs=[file_input, issue_description, car_type, OBD_codes], outputs=user_params_output, queue=False).then(bot, inputs=[auto_query, chatbot], outputs=[chatbot])

    clear_btn = gr.Button("Clear")

    #events
    msg.submit(user, inputs=[msg, chatbot], outputs=[chatbot], queue=False).then(bot, inputs=[msg, chatbot], outputs=[chatbot])
    msg.submit(lambda: "", outputs=[msg])
    clear_btn.click(lambda: None, None, chatbot, queue=False)

#launch interface
if __name__ == "__main__":
    app.launch()
