import streamlit as st
from summarizer import textrank_summarize

st.set_page_config(page_title="Tóm Tắt Văn Bản Tự Động", layout="wide")

st.title("📝 HỆ THỐNG TÓM TẮT VĂN BẢN TIẾNG VIỆT (PAGERANK / TEXTRANK)")
st.markdown("Đồ án môn học: Xử lý ngôn ngữ tự nhiên - Đề tài 6")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Nhập văn bản đầu vào")
    uploaded_file = st.file_uploader("Tải tệp văn bản (.txt)", type=["txt"])
    
    if uploaded_file is not None:
        input_text = uploaded_file.read().decode("utf-8")
    else:
        input_text = st.text_area("Hoặc dán văn bản vào đây:", height=300)
    
    st.subheader("Tham số cấu hình")
    num_sentences = st.number_input("Số lượng câu tóm tắt:", min_value=1, max_value=20, value=3)
    method = st.selectbox("Độ đo tương đồng:", ["Cosine", "Jaccard"])
    d_damping = st.slider("Hệ số d (Damping Factor):", min_value=0.1, max_value=0.99, value=0.85)

with col2:
    st.subheader("Kết quả tóm tắt")
    if st.button("🚀 Thực hiện tóm tắt", type="primary"):
        if not input_text.strip():
            st.warning("Vui lòng nhập nội dung văn bản!")
        else:
            summary, sentences, ranked_sentences = textrank_summarize(
                input_text, 
                num_sentences=num_sentences, 
                method=method.lower(), 
                d=d_damping
            )
            
            st.success("Tóm tắt hoàn tất!")
            st.write(summary)
            
            st.divider()
            st.subheader("📊 Bảng xếp hạng độ quan trọng các câu (PageRank)")
            for score, idx, sent in ranked_sentences:
                st.write(f"**[Điểm PR: {score:.4f}]** - *Câu {idx+1}*: {sent}")