from dotenv import load_dotenv
import os

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ChatMessageHistory
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

# ✅ 1. API 키 불러오기
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ✅ 2. 임베딩 및 벡터스토어 로드
embedding = OpenAIEmbeddings(model="text-embedding-3-large", openai_api_key=OPENAI_API_KEY)
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embedding)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# ✅ 3. LLM 정의
chat = ChatOpenAI(model="gpt-4o-mini", openai_api_key=OPENAI_API_KEY)

# ✅ 4. 프롬프트 템플릿 정의
question_answering_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "사용자의 질문에 대해 아래 context에 기반하여 답변하라. "
            "만약 context에 답이 없으면 '문서에 관련 정보가 없습니다.' 라고 답하라.\n\n{context}",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

# ✅ 5. 문서 → 답변 생성 체인 구성
document_chain = create_stuff_documents_chain(chat, question_answering_prompt)

# ✅ 6. 전체 RAG 체인 생성
rag_chain = create_retrieval_chain(retriever, document_chain)

# ✅ 7. 채팅 히스토리 객체 생성
chat_history = ChatMessageHistory()

# ✅ 8. 유저 질문
question = "토마토 뿔나방은 어떤 피해를 입히고, 어떻게 방제하나요?"
chat_history.add_user_message(question)

# ✅ 9. RAG 실행
response = rag_chain.invoke({
    "messages": chat_history.messages
})

# ✅ 10. 응답 저장 및 출력
chat_history.add_ai_message(response["answer"])
print("🧠 GPT 응답:")
print(response["answer"])
