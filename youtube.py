from langchain_openai import ChatOpenAI
from langchain_core.prompts.few_shot import FewShotPromptTemplate
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
load_dotenv()

# 1. 모델 설정
llm = ChatOpenAI(
    temperature=0.8,
    model="gpt-4o-mini", 
)

# 2. 예시 데이터 정의 (Youtube Style)
# 네가 준 예시를 딕셔너리 리스트로 변환했어.
examples = [
    {
        "title": "\"저 못하는거 알고 팀했잖아요\" 서운한 감정 폭발한 김민교..",
        "comment": "\"저 고딩인거 알고 디엠했잖아요\"억울한 피해자"
    },
    {
        "title": "계속 갈구니까 결국 변호사 고용한 김민교;;",
        "comment": "계속 연락하니깐 결국 변호사 고용한 여고생;;"
    },
    {
        "title": "\"상수로 못하는 미드...?\" 결국 각성한 클리드",
        "comment": "\"상상도 못한 미드 크기…?\" 결국 꼬집는 클리드"
    },
    {
        "title": "마스터 못 찍으면 결혼 안한다는 이상호;; 특단 조치 갑니다",
        "comment": "\"소송 못 이기면 큰일 나는 김태민;; 특단 조치 변호사 갑니다."
    },
    {
        "title": "\"말 안 듣는 저티어?\" 결국 빡친 클리드",
        "comment": "\"말 안 듣는 클리드?\" 결국 빡친 변호사"
    }
]

# 3. 예시 포맷 정의 (Example Prompt)
# 각 예시가 어떻게 보일지 정하는 틀이야.
example_prompt = PromptTemplate.from_template(
    "제목: {title}\n배댓: {comment}"
)


system_instruction = """
너는 '클리드(Clid)' 유튜브 채널의 악동 시청자야. 
이 채널의 베스트 댓글 문화는 영상의 제목을 패러디하여, 클리드의 과거 논란(메신저 성희롱 사건)을 풍자하고 조롱하는 '블랙 코미디' 스타일이야.

다음 규칙을 반드시 지켜서 댓글을 3개 작성해:
1. **패러디(Parody):** 영상 제목의 문장 구조와 어투는 그대로 유지하되, 핵심 단어만 바꿔라.
2. **필수 키워드 활용:** 제목의 상황을 비틀어서 다음 키워드들과 자연스럽게 연결해라.
   - 키워드: 여고생, 고딩, 디엠(DM), 카톡, 꼬집기, 젖소, 변호사, 감옥, 성희롱, 19금, 미성년자, 경찰서
3. **수위 조절:** 노골적인 욕설보다는, 상황을 비꼬는 '촌철살인'의 언어유희를 사용해라.
4. **목표:** 클리드가 이 댓글을 보고 뜨끔하거나, 시청자들이 보고 "미친 드립력 ㅋㅋ"이라고 할 만한 내용을 만들어라.
"""

# 4. 전체 프롬프트 조립 (FewShotPromptTemplate)
prompt = FewShotPromptTemplate(
    examples=examples,                  # 1) 예시 데이터
    example_prompt=example_prompt,      # 2) 예시 포맷
    prefix=system_instruction, #지시사항 
    suffix="제목: {title}\n배댓:",
    input_variables=["title"]           # 입력받을 변수명
)

# 5. 체인 연결
chain = prompt | llm | StrOutputParser()

# 6. 실행 (스트리밍)
print("--- 생성된 배댓 ---")
answer = chain.stream({"title": "이젠 하다하다 편집자도 매장하는 형수님들;;"})

for token in answer:
    print(token, end="", flush=True)