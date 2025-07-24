# rag_api.py
from fastapi import FastAPI, Request
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ChatMessageHistory
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

# 환경 변수 로드
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# FastAPI 앱 초기화
app = FastAPI()

# CORS 설정 (Spring에서 호출 가능하게)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 또는 ["http://localhost:8080"] 등으로 제한 가능
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# LangChain 설정
embedding = OpenAIEmbeddings(model="text-embedding-3-large", openai_api_key=OPENAI_API_KEY)
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embedding)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
chat = ChatOpenAI(model="gpt-4o-mini", openai_api_key=OPENAI_API_KEY)

question_answering_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "사용자의 질문에 대해 아래 context에 기반하여 답변하라. "
        "만약 context에 답이 없으면 '문서에 관련 정보가 없습니다.' 라고 답하라.\n\n{context}",
    ),
    MessagesPlaceholder(variable_name="messages"),
])
document_chain = create_stuff_documents_chain(chat, question_answering_prompt)
rag_chain = create_retrieval_chain(retriever, document_chain)

# 메시지 기록 객체 (간단하게 세션 저장 없이 단발성 구현)
chat_history = ChatMessageHistory()

# 요청 스키마
class Question(BaseModel):
    question: str

@app.post("/ask")
async def ask_question(data: Question):
    question = data.question
    chat_history.add_user_message(question)

    response = rag_chain.invoke({
        "input": question,
        "messages": chat_history.messages
    })

    chat_history.add_ai_message(response["answer"])
    return {"answer": response["answer"]}
