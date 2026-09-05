import streamlit as st
from datetime import datetime
from summarizer import textrank_summarize
import docx
from pypdf import PdfReader
import chromadb

# Khởi tạo Vector Database ChromaDB (lưu dữ liệu cục bộ dạng Persistent)
@st.cache_resource
def get_chroma_client():
    return chromadb.PersistentClient(path="./chroma_db")

client = get_chroma_client()
collection = client.get_or_create_collection(name="summary_history")

def save_to_chroma(original_text, summary_text, method, num_sentences):
    doc_id = f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    collection.add(
        documents=[summary_text],
        metadatas=[{
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "original_text": original_text[:500], # Lưu xem trước 500 ký tự văn bản gốc
            "method": method,
            "num_sentences": num_sentences
        }],
        ids=[doc_id]
    )

def extract_text_from_file(uploaded_file):
    file_type = uploaded_file.name.split('.')[-1].lower()
    text = ""
    if file_type == "txt":
        text = uploaded_file.read().decode("utf-8", errors="ignore")
    elif file_type == "docx":
        doc = docx.Document(uploaded_file)
        text = "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
    elif file_type == "pdf":
        reader = PdfReader(uploaded_file)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

st.set_page_config(page_title="Tóm Tắt Văn Bản Tự Động", layout="wide")
st.title("📝 HỆ THỐNG TÓM TẮT VĂN BẢN TIẾNG VIỆT (PAGERANK / TEXTRANK)")
st.markdown("Đồ án môn học: Xử lý ngôn ngữ tự nhiên - Đề tài 6")

tab1, tab2 = st.tabs(["🚀 Thực hiện tóm tắt", "📜 Lịch sử CSDL Vector (ChromaDB)"])

with tab1:
    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("Nhập văn bản đầu vào")
        uploaded_file = st.file_uploader("Tải tệp văn bản (.txt, .docx, .pdf)", type=["txt", "docx", "pdf"])
        
        input_text = ""
        if uploaded_file is not None:
            input_text = extract_text_from_file(uploaded_file)
            st.info(f"Đã đọc xong file **{uploaded_file.name}** ({len(input_text)} ký tự)")
            with st.expander("Xem trước nội dung tệp"):
                st.write(input_text)
        else:
            input_text = st.text_area("Hoặc dán văn bản vào đây:", height=250)
        
        st.subheader("Tham số cấu hình")
        num_sentences = st.number_input("Số lượng câu tóm tắt:", min_value=1, max_value=20, value=3)
        method = st.selectbox("Độ đo tương đồng:", ["Cosine", "Jaccard"])
        d_damping = st.slider("Hệ số d (Damping Factor):", min_value=0.1, max_value=0.99, value=0.85)

    with col2:
        st.subheader("Kết quả tóm tắt")
        if st.button("🚀 Thực hiện tóm tắt", type="primary"):
            if not input_text.strip():
                st.warning("Vui lòng nhập nội dung văn bản hoặc tải file lên!")
            else:
                summary, sentences, ranked_sentences = textrank_summarize(
                    input_text, num_sentences=num_sentences, method=method.lower(), d=d_damping
                )
                # Lưu lịch sử vào Vector DB ChromaDB
                save_to_chroma(input_text, summary, method, num_sentences)
                
                st.success("Tóm tắt hoàn tất & Đã lưu vào Vector Database (ChromaDB)!")
                st.write(summary)
                
                st.divider()
                st.subheader("📊 Bảng xếp hạng điểm PageRank các câu")
                for score, idx, sent in ranked_sentences:
                    st.write(f"**[Điểm PR: {score:.4f}]** - *Câu {idx+1}*: {sent}")

with tab2:
    st.subheader("Lịch sử các lần tóm tắt lưu trong Vector Database (ChromaDB)")
    results = collection.get()
    
    if not results or not results["ids"]:
        st.info("Chưa có dữ liệu lịch sử trong ChromaDB.")
    else:
        for idx in range(len(results["ids"])):
            doc_id = results["ids"][idx]
            summary_content = results["documents"][idx]
            meta = results["metadatas"][idx]
            
            with st.expander(f"Mã lưu trữ: {doc_id} | Thời gian: {meta.get('created_at')} | Độ đo: {meta.get('method')}"):
                st.markdown(f"**Nội dung tóm tắt:**\n{summary_content}")
                st.caption(f"Trích văn bản gốc: {meta.get('original_text')}...")