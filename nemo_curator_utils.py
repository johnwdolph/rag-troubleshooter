import json
from llama_index.core import VectorStoreIndex, StorageContext, Document, SimpleDirectoryReader, Settings
from nemo_curator.download import download_common_crawl, download_wikipedia, ResiliparseExtractor, JusTextExtractor
from nemo_curator import ScoreFilter, Sequential
from nemo_curator.modifiers.unicode_reformatter import UnicodeReformatter
from nemo_curator.modules.modify import Modify
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.nvidia import NVIDIAEmbedding
from llama_index.vector_stores.milvus import MilvusVectorStore
from multiprocessing import freeze_support

import time
import os

class CuratorPipeline:
    curation_pipeline = Sequential([
        # Fix unicode
        Modify(UnicodeReformatter()),
        # # Discard short records
        # ScoreFilter(WordCountFilter(min_words=80)),
        # # Discard low-quality records
        # ScoreFilter(FastTextQualityFilter(model_path="model.bin")),
        # # Discard records from the evaluation metrics to prevent test set leakage.
        # TaskDecontamination([Winogrande(), Squad(), TriviaQA()])
    ])

class CommonCrawl:

    Settings.text_splitter = SentenceSplitter(chunk_size=500)
    Settings.embed_model = NVIDIAEmbedding(mode="NV-Embed-QA", truncate="END")

    def __init__(self, index_path='./vectorstores/milvus_store_curator.db', url_limit=5):
        self.vector_store = MilvusVectorStore(uri=index_path, dim=1024, overwrite=True)
        self.storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
        self.curator_index = None
        self.index_path = index_path
        self.index = VectorStoreIndex([])
        self.url_limit=url_limit
        
        self.curator_index = self.load_index_from_disk()

    def get_curator_index(self):
        # if not self.curator_index:
        # freeze_support()
        self.start_download()
        return self.curator_index

    def load_index_from_disk(self):
        if os.path.exists(self.index_path):
            documents = SimpleDirectoryReader(input_files=[self.index_path]).load_data()
            return VectorStoreIndex.from_documents(documents)
        else:
            return None

    def start_download(self):
        print('Running first-time curation downloads...')
        extraction_algorithms = [ResiliparseExtractor(), JusTextExtractor()]

        # Download and sample data
        common_crawl = download_common_crawl(
            './datasets/',
            "2022-05",
            "2024-04",
            output_type='jsonl',
            url_limit=self.url_limit,
            algorithm=extraction_algorithms[0]
        )

        if common_crawl:# and not common_crawl.empty:
            #process data with curator pipeline
            common_crawl_processed = CuratorPipeline.curation_pipeline(common_crawl)

            # Assuming common_crawl.df is a DataFrame or similar object
            sample = common_crawl_processed.df.sample(frac=10 / len(common_crawl_processed.df))

            self.add_to_curator_index(sample)

    def add_to_curator_index(self, data_dataframe):
        docs = [Document(text=row['text']) for _, row in data_dataframe.iterrows()]
        self.curator_index = VectorStoreIndex.from_documents(docs, storage_context=self.storage_context)