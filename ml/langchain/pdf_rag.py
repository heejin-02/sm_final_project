from dotenv import load_dotenv
import os

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

# 환경 변수 로드
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# 임베딩 모델 설정
embedding = OpenAIEmbeddings(model="text-embedding-3-large", openai_api_key=OPENAI_API_KEY)

# 벡터스토어 로드
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embedding)

# ✅ PDF 추가 함수
def load_and_add_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    splits = splitter.split_documents(docs)

    vectorstore.add_documents(splits)
    print(f"✅ '{pdf_path}' 내용이 벡터스토어에 추가되었습니다.")

# ✅ 질문 함수
def ask_question(question: str):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    chat = ChatOpenAI(model="gpt-4o-mini", openai_api_key=OPENAI_API_KEY)

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "사용자의 질문에 대해 아래 context에 기반하여 답변하라. "
            "만약 context에 답이 없으면 '문서에 관련 정보가 없습니다.' 라고 답하라.\n\n{context}",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ])

    document_chain = create_stuff_documents_chain(chat, prompt)
    rag_chain = create_retrieval_chain(retriever, document_chain)

    chat_history = ChatMessageHistory()
    chat_history.add_user_message(question)

    response = rag_chain.invoke({
        "input": question,
        "messages": chat_history.messages
    })

    chat_history.add_ai_message(response["answer"])
    print("🧠 GPT 응답:")
    print(response["answer"])


# ✅ 실행 예시
if __name__ == "__main__":
    # load_and_add_pdf("./pdf/2010병해충관리.pdf")
    ask_question("그 병에 대해 설명하세요")
