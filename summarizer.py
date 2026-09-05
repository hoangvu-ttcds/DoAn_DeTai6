import re
import numpy as np
import networkx as nx
from underthesea import word_tokenize

VIETNAMESE_STOPWORDS = set([
    "và", "hoặc", "nhưng", "vì", "nên", "bởi", "tại", "trong", "ngoài", "trên",
    "dưới", "những", "các", "mọi", "mỗi", "một", "hai", "là", "có", "được", "bị",
    "cho", "với", "theo", "đến", "từ", "này", "đó", "khi", "như", "để", "ra"
])

# BƯỚC 1: TÁCH CÂU
def step1_split_sentences(text):
    sentences = re.split(r'(?<=[.!?])\s+|\n+', text)
    return [s.strip() for s in sentences if len(s.strip()) > 10]

# BƯỚC 2, 3, 4: TÁCH TỪ -> TIỀN XỬ LÝ LÀM SẠCH -> LOẠI BỎ TỪ DỪNG
def preprocess_sentence(sentence):
    tok_sent = word_tokenize(sentence, format="text")  # B2: Tách từ
    lowered = tok_sent.lower()                         # B3: Tiền xử lý chữ thường
    cleaned = re.sub(r'[^\w\s_]', '', lowered)        # B3: Xóa dấu câu
    words = cleaned.split()
    final_words = [w for w in words if w not in VIETNAMESE_STOPWORDS and w.isalnum()] # B4: Lọc từ dừng
    return final_words

def cosine_similarity(tokens1, tokens2):
    vocab = list(set(tokens1) | set(tokens2))
    if not vocab:
        return 0.0
    v1 = np.array([tokens1.count(w) for w in vocab])
    v2 = np.array([tokens2.count(w) for w in vocab])
    norm = np.linalg.norm(v1) * np.linalg.norm(v2)
    return np.dot(v1, v2) / norm if norm > 0 else 0.0

def jaccard_similarity(tokens1, tokens2):
    set1, set2 = set(tokens1), set(tokens2)
    if not set1 or not set2:
        return 0.0
    return len(set1 & set2) / float(len(set1 | set2))

def build_similarity_matrix(sentences, method="cosine"):
    processed_sentences = [preprocess_sentence(s) for s in sentences]
    n = len(sentences)
    sim_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                if method == "jaccard":
                    sim_matrix[i][j] = jaccard_similarity(processed_sentences[i], processed_sentences[j])
                else:
                    sim_matrix[i][j] = cosine_similarity(processed_sentences[i], processed_sentences[j])
    return sim_matrix

def textrank_summarize(text, num_sentences=3, method="cosine", d=0.85):
    sentences = step1_split_sentences(text)
    if len(sentences) <= num_sentences:
        return text, sentences, []
    
    sim_matrix = build_similarity_matrix(sentences, method=method)
    nx_graph = nx.from_numpy_array(sim_matrix)
    scores = nx.pagerank(nx_graph, alpha=d)
    
    ranked_sentences = sorted(((scores[i], i, s) for i, s in enumerate(sentences)), reverse=True)
    top_sentences = sorted(ranked_sentences[:num_sentences], key=lambda x: x[1])
    summary = " ".join([sent for score, idx, sent in top_sentences])
    
    return summary, sentences, ranked_sentences
