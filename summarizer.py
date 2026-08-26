import re
import numpy as np
import networkx as nx
from underthesea import sent_tokenize, word_tokenize

# 1. Danh sách từ dừng tiếng Việt cơ bản
STOP_WORDS = set([
    "và", "của", "các", "có", "được", "cho", "trong", "về", "với", "như", "là", 
    "đã", "đang", "sẽ", "không", "khi", "tôi", "này", "đó", "đến", "từ", "một", 
    "những", "cũng", "để", "ra", "bị", "theo", "tại", "nhiều", "rất", "hơn", 
    "mình", "người", "nhưng", "vẫn", "vì", "nên", "sau", "cả", "lại", "thì", 
    "khác", "mọi", "cần", "ngay", "qua", "lên", "bằng", "hoặc", "nếu", "sự", 
    "việc", "thêm", "cùng", "luôn", "chỉ", "còn", "gì", "nào", "đâu", "ai"
])

# 2. Tách câu và tiền xử lý tách từ, loại bỏ từ dừng
def preprocess_text(text):
    sentences = sent_tokenize(text)
    processed_sentences = []
    
    for sent in sentences:
        segmented = word_tokenize(sent.lower(), format="text")
        words = re.findall(r'\b[a-zàáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ_]+\b', segmented)
        words_clean = [w for w in words if w not in STOP_WORDS and len(w) > 1]
        processed_sentences.append(words_clean)
        
    return sentences, processed_sentences

# 3. Tính độ tương đồng Jaccard giữa 2 câu
def jaccard_similarity(words1, words2):
    set1, set2 = set(words1), set(words2)
    if not set1 or not set2:
        return 0.0
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    return float(len(intersection)) / float(len(union))

# 4. Tính độ tương đồng Cosine giữa 2 câu
def cosine_similarity(words1, words2):
    set1, set2 = set(words1), set(words2)
    all_words = list(set1.union(set2))
    if not all_words:
        return 0.0
    
    v1 = [1 if w in set1 else 0 for w in all_words]
    v2 = [1 if w in set2 else 0 for w in all_words]
    
    dot_product = sum(a * b for a, b in zip(v1, v2))
    norm_v1 = sum(a * a for a in v1) ** 0.5
    norm_v2 = sum(b * b for b in v2) ** 0.5
    
    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0
    return dot_product / (norm_v1 * norm_v2)

# 5. Khởi tạo Ma trận tương đồng (Similarity Matrix)
def build_similarity_matrix(processed_sentences, method="cosine"):
    n = len(processed_sentences)
    sim_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            if i != j:
                if method == "jaccard":
                    sim_matrix[i][j] = jaccard_similarity(processed_sentences[i], processed_sentences[j])
                else:
                    sim_matrix[i][j] = cosine_similarity(processed_sentences[i], processed_sentences[j])
                    
    return sim_matrix

# 6. Thuật toán PageRank và trích xuất câu tóm tắt
def textrank_summarize(text, num_sentences=3, method="cosine", d=0.85):
    sentences, processed_sentences = preprocess_text(text)
    
    if len(sentences) <= num_sentences:
        return text, sentences, []
    
    sim_matrix = build_similarity_matrix(processed_sentences, method=method)
    
    nx_graph = nx.from_numpy_array(sim_matrix)
    scores = nx.pagerank(nx_graph, alpha=d)
    
    ranked_sentences = sorted(((scores[i], i, s) for i, s in enumerate(sentences)), reverse=True)
    
    selected_indices = sorted([item[1] for item in ranked_sentences[:num_sentences]])
    summary = " ".join([sentences[i] for i in selected_indices])
    
    return summary, sentences, ranked_sentences