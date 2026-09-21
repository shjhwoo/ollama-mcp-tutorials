import logging

from langchain_ollama import OllamaEmbeddings
from ollama import ResponseError
from sklearn.metrics.pairwise import cosine_similarity

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

EMBEDDING_UNSUPPORTED_NOTICE = "{model} 모델은 임베딩을 지원하지 않아 유사도 비교에서 제외합니다."

# 1. 올라마 임베딩 모델들을 사용할 수 있도록 준비
bge_embed = OllamaEmbeddings(model="bge-m3")
nomic_embed = OllamaEmbeddings(model="nomic-embed-text")
llm_embed = OllamaEmbeddings(model="qwen3:8b")

# 2. 사용자로부터 문장 3개를 입력 받음
sentences = [input(f"문장 {i + 1}을 입력하세요: ") for i in range(3)]


def embed_sentences(embeddings: OllamaEmbeddings, sentences: list[str]) -> list[list[float]] | None:
    """문장들을 임베딩 벡터로 변환한다. 모델이 임베딩을 지원하지 않으면 None을 반환한다."""
    try:
        return [embeddings.embed_query(sentence) for sentence in sentences]
    except ResponseError as error:
        logger.error("%s 모델 임베딩 생성 실패: %s", embeddings.model, error)
        print(EMBEDDING_UNSUPPORTED_NOTICE.format(model=embeddings.model))
        return None


# 3. 임베딩 모델 별로 문장 3개에 대한 임베딩 벡터 추출
bge_vectors = embed_sentences(bge_embed, sentences)
nomic_vectors = embed_sentences(nomic_embed, sentences)
llm_vectors = embed_sentences(llm_embed, sentences)

# 4. 문장 벡터 간 코사인 유사도 계산 (임베딩 생성에 성공한 모델만 계산)
bge_similarities = cosine_similarity(bge_vectors) if bge_vectors is not None else None
nomic_similarities = cosine_similarity(nomic_vectors) if nomic_vectors is not None else None
llm_similarities = cosine_similarity(llm_vectors) if llm_vectors is not None else None

# 5. 문장 간 유사도를 임베딩 모델 별로 화면에 출력
for i in range(len(sentences)):
    for j in range(i + 1, len(sentences)):
        print(f"\n- 문장 {i + 1}과 문장 {j + 1}의 유사도 -")
        if bge_similarities is not None:
            print(f"BGE-M3: {bge_similarities[i][j]:.2f}")
        if nomic_similarities is not None:
            print(f"Nomic-embed-text: {nomic_similarities[i][j]:.2f}")
        if llm_similarities is not None:
            print(f"QWEN3: {llm_similarities[i][j]:.2f}")
