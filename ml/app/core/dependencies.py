"""
의존성 주입 관리
"""
from functools import lru_cache
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from openai import OpenAI
import requests

from app.core.config import settings

class Dependencies:
    def __init__(self):
        self._openai_client = None
        self._chat_openai = None
        self._embeddings = None
        self._vectorstore = None
        
    @property
    def openai_client(self) -> OpenAI:
        if not self._openai_client:
            self._openai_client = OpenAI(api_key=settings.OPENAI_API_KEY)
        return self._openai_client
    
    @property 
    def chat_openai(self) -> ChatOpenAI:
        if not self._chat_openai:
            self._chat_openai = ChatOpenAI(
                model=settings.GPT_MODEL,
                openai_api_key=settings.OPENAI_API_KEY
            )
        return self._chat_openai
    
    @property
    def embeddings(self) -> OpenAIEmbeddings:
        if not self._embeddings:
            self._embeddings = OpenAIEmbeddings(
                model=settings.EMBEDDING_MODEL,
                openai_api_key=settings.OPENAI_API_KEY
            )
        return self._embeddings
    
    @property
    def vectorstore(self) -> Chroma:
        if not self._vectorstore:
            self._vectorstore = Chroma(
                persist_directory=settings.CHROMA_DB_PATH,
                embedding_function=self.embeddings
            )
        return self._vectorstore

@lru_cache()
def get_dependencies() -> Dependencies:
    return Dependencies()

def get_openai_client() -> OpenAI:
    return get_dependencies().openai_client

def get_chat_openai() -> ChatOpenAI:
    return get_dependencies().chat_openai

def get_embeddings() -> OpenAIEmbeddings:
    return get_dependencies().embeddings

def get_vectorstore() -> Chroma:
    return get_dependencies().vectorstore