import streamlit as st
import sqlite3
from datetime import datetime
from summarizer import textrank_summarize

def init_db():
    conn = sqlite3.connect('history.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS summary_history 
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, 
                  created_at TEXT, 
                  original_text TEXT, 
                  summary_text TEXT, 
                  method TEXT, 
                  num_sentences INTEGER)''')
    conn.commit()
    conn.close()

def save_history(original_text, summary_text, method, num_sentences):
    conn = sqlite3.connect('history.db')
    c = conn.cursor()
    c.execute('''INSERT INTO summary_history 
                 (created_at, original_text, summary_text, method, num_sentences) 
                 VALUES (?, ?, ?, ?, ?)''', 
              (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), original_text, summary_text, method, num_sentences))
    conn.commit()
    conn.close()

init_db()

st.set_page_config(page_title="Tóm Tắt Văn Bản Tự Động", layout="wide")
st.title("📝 HỆ THỐNG TÓM TẮT VĂN BẢN TIẾNG VIỆT (PAGERANK / TEXTRANK)")
st.markdown("Đồ án môn học: Xử lý ngôn ngữ tự nhiên - Đề tài 6")

tab1, tab2 = st.tabs(["🚀 Thực hiện tóm tắt", "📜 Lịch sử CSDL (SQLite)"])

with tab1:
    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("Nhập văn bản đầu vào")
        uploaded_file = st.file_uploader("Tải tệp văn bản (.txt)", type=["txt"])
        if uploaded_file is not None:
            input_text = uploaded_file.read().decode("utf-8")
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
                st.warning("Vui lòng nhập nội dung văn bản!")
            else:
                summary, sentences, ranked_sentences = textrank_summarize(
                    input_text, num_sentences=num_sentences, method=method.lower(), d=d_damping
                )
                save_history(input_text, summary, method, num_sentences)
                
                st.success("Tóm tắt hoàn tất & Đã lưu lịch sử vào CSDL!")
                st.write(summary)
                
                st.divider()
                st.subheader("📊 Bảng xếp hạng điểm PageRank các câu")
                for score, idx, sent in ranked_sentences:
                    st.write(f"**[Điểm PR: {score:.4f}]** - *Câu {idx+1}*: {sent}")

with tab2:
    st.subheader("Lịch sử các lần tóm tắt lưu trong SQLite")
    conn = sqlite3.connect('history.db')
    c = conn.cursor()
    c.execute("SELECT id, created_at, method, num_sentences, summary_text FROM summary_history ORDER BY id DESC")
    data = c.fetchall()
    conn.close()
    
    if not data:
        st.info("Chưa có dữ liệu lịch sử.")
    else:
        for row in data:
            with st.expander(f"Lần tóm tắt #{row[0]} | Thời gian: {row[1]} | Độ đo: {row[2]} | Số câu: {row[3]}"):
                st.write(row[4])