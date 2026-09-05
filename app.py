@st.cache_resource
def get_chroma_client():
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_or_create_collection(name="summary_history")
    
    # Nếu CSDL rỗng, tự động nạp 1 dữ liệu mẫu mặc định
    if collection.count() == 0:
        collection.add(
            documents=["Trí tuệ nhân tạo (AI) đang phát triển vô cùng mạnh mẽ. Xử lý ngôn ngữ tự nhiên giúp máy tính hiểu và phân tích ngôn ngữ con người."],
            metadatas=[{
                "created_at": "2026-09-05 10:00:00",
                "original_text": "Trí tuệ nhân tạo (AI) đang phát triển vô cùng mạnh mẽ trên toàn thế giới. Các công nghệ tiên tiến như Học máy và Học sâu...",
                "method": "Cosine",
                "num_sentences": 2
            }],
            ids=["sample_summary_01"]
        )
    return client
