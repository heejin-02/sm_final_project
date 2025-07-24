from dotenv import load_dotenv
import os

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

# ✅ 1. 환경변수 로드 및 API 키 설정
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ✅ 2. 벡터스토어 초기화 (기존 임베딩 사용)
embedding = OpenAIEmbeddings(model="text-embedding-3-large", openai_api_key=OPENAI_API_KEY)
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embedding)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# ✅ 3. LLM 모델 정의
chat = ChatOpenAI(model="gpt-4o", openai_api_key=OPENAI_API_KEY)

# ✅ 4. 프롬프트 정의
question_answering_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "사용자의 질문에 대해 아래 context에 기반하여 친환경 방제법을 중심으로 설명하라. "
        "문서에 관련 정보가 없으면 '문서에 관련 정보가 없습니다.' 라고만 답하라.\n\n{context}"
    ),
    MessagesPlaceholder(variable_name="messages")
])

# ✅ 5. 문서 → 답변 생성 체인 구성
document_chain = create_stuff_documents_chain(chat, question_answering_prompt)

# ✅ 6. RAG 전체 체인 구성
rag_chain = create_retrieval_chain(retriever, document_chain)

# ✅ 7. 채팅 히스토리 객체 생성
chat_history = ChatMessageHistory()

# ✅ 8. 함수 정의
def generate_report_from_insect(insect_name: str) -> str:
    chat_history.add_user_message(insect_name + " 방제 방법 알려줘")
    response = rag_chain.invoke({
        "input": insect_name,
        "messages": chat_history.messages
    })
    chat_history.add_ai_message(response["answer"])
    return response["answer"]

# ✅ 실행 예시
if __name__ == "__main__":
    # PDF 벡터스토어 추가는 1회만 필요. 이후 주석 처리
    # loader = PyPDFLoader("./pdf/2010병해충관리.pdf")
    # pages = loader.load_and_split()
    # vectorstore.add_documents(pages)
    # vectorstore.persist()

    # 예시 질문 (리포트에서 가장 많이 등장한 해충 이름 넣기)
    report = generate_report_from_insect("담배가루이")
    print("\n✅ GPT 방제 리포트:\n")
    print(report)
