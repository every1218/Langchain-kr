# langchain-kr-study/test.py

# ##############################################################################
#
# 이 스크립트는 langchain-kr 레포지토리의 핵심 기능을 요약한 RAG(Retrieval-Augmented Generation) 파이프라인 예제입니다.
#
# 기능:
# 1. PDF 문서 로드
# 2. 텍스트를 청크(chunk) 단위로 분할
# 3. OpenAI 임베딩 모델을 사용하여 각 청크를 벡터로 변환
# 4. FAISS 벡터 저장소에 벡터 저장
# 5. 질문에 가장 유사한 청크를 검색하는 검색기(retriever) 생성
# 6. LLM(gpt-3.5-turbo)을 사용하여 검색된 컨텍스트를 기반으로 질문에 답변
#
# 실행 전 필수 설정:
#
# 1. 필요 라이브러리 설치:
#    pip install -qU python-dotenv langchain-openai langchain-community langchain-text-splitters faiss-cpu pymupdf
#
# 2. PDF 파일 준비:
#    - 다운로드 링크: https://spri.kr/posts/view/23669
#    - 위 링크에서 'SPRI_AI_Brief_2023년12월호_F.pdf' 파일을 다운로드합니다.
#    - 레포지토리 루트에 '12-RAG/data' 폴더를 생성합니다.
#    - 다운로드한 PDF 파일을 '12-RAG/data/' 폴더에 저장합니다.
#      (전체 경로: c:/Users/DOHYEON/Documents/langchain-kr/12-RAG/data/SPRI_AI_Brief_2023년12월호_F.pdf)
#
# 3. OpenAI API 키 설정:
#    - 레포지토리 루트에 '.env' 파일을 생성합니다.
#    - 파일 내용: OPENAI_API_KEY="your_openai_api_key"
#
# ##############################################################################

import os
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

def main():
    """
    RAG 파이프라인을 실행하는 메인 함수
    """
    # API 키 로드
    load_dotenv()

    # --- 1. PDF 파일 경로 확인 ---
    pdf_path = "12-RAG/data/SPRI_AI_Brief_2023년12월호_F.pdf"
    if not os.path.exists(pdf_path):
        print(f"오류: PDF 파일을 찾을 수 없습니다. '{pdf_path}'")
        print("스크립트 상단의 '실행 전 필수 설정'을 확인해주세요.")
        return

    # --- 2. RAG 파이프라인 구축 ---
    try:
        # 단계 1: 문서 로드 (Load Documents)
        print("1단계: PDF 문서 로드 중...")
        loader = PyMuPDFLoader(pdf_path)
        docs = loader.load()
        print(f"  - 총 {len(docs)} 페이지의 문서를 로드했습니다.")

        # 단계 2: 문서 분할 (Split Documents)
        print("2단계: 문서를 청크로 분할 중...")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
        split_documents = text_splitter.split_documents(docs)
        print(f"  - 문서를 {len(split_documents)}개의 청크로 분할했습니다.")

        # 단계 3: 임베딩 생성 (Create Embedding)
        print("3단계: OpenAI 임베딩 모델 로드 중...")
        embeddings = OpenAIEmbeddings()

        # 단계 4: 벡터 DB 생성 및 저장 (Create DB & Store)
        print("4단계: FAISS 벡터 저장소 생성 중...")
        vectorstore = FAISS.from_documents(documents=split_documents, embedding=embeddings)
        print("  - 벡터 저장소 생성이 완료되었습니다.")

        # 단계 5: 검색기 생성 (Create Retriever)
        print("5단계: 벡터 저장소로부터 검색기 생성 중...")
        retriever = vectorstore.as_retriever()
        print("  - 검색기 생성이 완료되었습니다.")

        # 단계 6: 프롬프트 생성 (Create Prompt)
        prompt = PromptTemplate.from_template(
            """당신은 질문-답변(Question-Answering)을 수행하는 AI 어시스턴트입니다.
            제공된 컨텍스트(context)를 사용하여 질문(question)에 답변해 주세요.
            만약 답을 모른다면, 모른다고 답변하세요. 답변은 한국어로 작성해 주세요.

            #Context:
            {context}

            #Question:
            {question}

            #Answer:"""
        )

        # 단계 7: 언어 모델 생성 (Create LLM)
        llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

        # 단계 8: 체인 생성 (Create Chain)
        chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

        # --- 3. 체인 실행 ---
        print("\n--- RAG 파이프라인 실행 ---")
        question = "삼성전자가 자체 개발한 AI 의 이름은?"
        print(f"질문: {question}")

        response = chain.invoke(question)

        print("\n답변:")
        print(response)

    except Exception as e:
        print(f"\n오류가 발생했습니다: {e}")
        print("OpenAI API 키가 올바르게 설정되었는지, 필요한 라이브러리가 모두 설치되었는지 확인해주세요.")


if __name__ == "__main__":
    main()
