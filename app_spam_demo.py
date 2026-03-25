# ============================================================
# DEMO APP: Nhận Diện Spam Tiếng Việt (CLASSIC ML ONLY - FULL CHARTS)
# ============================================================
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import re
import os
import scipy.sparse as sp
import matplotlib.pyplot as plt
from pyvi import ViTokenizer
from gensim.models import Word2Vec

st.set_page_config(page_title="Hệ thống lọc Spam Tiếng Việt", page_icon="🛡️", layout="wide")

st.markdown("""
<style>
.main-title{font-size:2.2rem;font-weight:800;background:linear-gradient(90deg,#e74c3c,#3498db);-webkit-background-clip:text;-webkit-text-fill-color:transparent;text-align:center;margin-bottom:1rem;}
.spam-box{background:linear-gradient(135deg,#ff6b6b,#ee5a24);color:white;padding:1.5rem 2rem;border-radius:12px;text-align:center;font-size:1.8rem;font-weight:800;box-shadow:0 4px 20px rgba(238,90,36,.4)}
.ham-box{background:linear-gradient(135deg,#55efc4,#00b894);color:white;padding:1.5rem 2rem;border-radius:12px;text-align:center;font-size:1.8rem;font-weight:800;box-shadow:0 4px 20px rgba(0,184,148,.4)}
</style>""", unsafe_allow_html=True)

# ── 1. ĐƯỜNG DẪN TUYỆT ĐỐI ──
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
STOPWORDS_PATH = os.path.join(BASE_DIR, "vietnamese-stopwords.txt")

SAMPLES = {
    # ================= 🔴 10 MẪU SPAM (TỪ DỄ ĐẾN KHÓ) =================
    "🔴 Spam 1 (Bán hàng)": "Call: 0869758899 E thanh li'' 0822496000 còn 499k.Ba'n Them 0858999221 dong gia 499k",
    "🔴 Spam 2 (Lừa đảo/Cho vay)": "Vay vốn ngân hàng KHÔNG CẦN THẾ CHẤP, giải ngân trong 30p. LH ngay 0988xxxxxx",
    "🔴 Spam 3 (Giả danh người quen)": "Hoàng ơi, t đổi số rồi nhé. M lưu số này nha, số kia t bỏ. Chiều mượn tạm 500k đóng tiền trọ mai gửi lại được k?",
    "🔴 Spam 4 (Tuyển dụng trá hình)": "Dạ chào anh/chị, em là Thủy bên NS công ty TNHH Vạn Phát. Thấy CV của mình phù hợp, bên em mời anh/chị làm CTV online tại nhà, ngày làm 2 tiếng nhận lương ngày. IB zalo em nhé.",
    "🔴 Spam 5 (Giả mạo giao hàng)": "Shipper J&T: Don hang 89123 cua ban giao khong thanh cong do thieu thong tin. Vui long bo sung tai: htps://jnt-vn.com de nhan hang.",
    "🔴 Spam 6 (Lừa đảo chuyển nhầm)": "Cháu ơi cô chuyển nhầm 2 triệu vào tài khoản cháu lúc nãy. Cháu kiểm tra rồi chuyển lại giúp cô vào stk 0123456789 ngân hàng MB nhé. Cô cảm ơn cháu.",
    "🔴 Spam 7 (Mạo danh Cơ quan nhà nước)": "Cuc Thue VN: Canh bao nop phat cham thue TNCN nam 2025. Chi tiet tai ung dung eTax hoac truy cap thuevietnam-gov.com. Bo qua tin nhan neu da nop.",
    "🔴 Spam 8 (Đầu tư tài chính/Coin)": "Chuc mung ban dc chon tham gia room VIP ho tro chung khoan thuc chien. Loi nhuan 30%/thang. Add Zalo Truong phong: 0912xxxxxx de vao nhom.",
    "🔴 Spam 9 (Trúng thưởng nền tảng)": "Shoppee: Tai khoan cua ban nhan duoc 1 phan qua tri an tri gia 3.000.000 VND. Click vao link :https://www.youtube.com/watch?v=6Hv80f1-9UQde xac nhan thong tin nhan thuong.",
    "🔴 Spam 10 (Gia hạn dịch vụ giả)": "Cảnh báo: Tên miền của bạn sẽ hết hạn trong 24h tới. Vui lòng thanh toán phí duy trì 550k vào STK 123456789 (Nguyen Van A) để tránh gián đoạn dịch vụ.",

    # ================= 🟢 10 MẪU HAM (TÌNH HUỐNG THỰC TẾ/GÂY NHIỄU) =================
    "🟢 Ham 1 (Bóc phốt)": "toi viết những dòng này sau khi bị nhóm intro to ai này scam 500k, cụ thể họ đã bắt tôi trả tiền...",
    "🟢 Ham 2 (Chê dịch vụ)": "phí dịch vụ cao nhất , chuyển tiền nội mạng trừ phí tởm nhất , thái độ làm việc bố láo nhất",
    "🟢 Ham 3 (Đời thường)": "Chuyến e đi Ninh Bình với ny có 2 triệu thôi =))",
    "🟢 Ham 4 (Nhắc chuyển khoản)": "Ê t vừa chuyển khoản 2 củ tiền vé máy bay rồi nha. Check app ngân hàng xem nhận được chưa, không thấy báo tao để tao gọi tổng đài.",
    "🟢 Ham 5 (Share link săn sale)": "Mày lên Shopee săn ngay cái voucher freeship 50k đi, đang sale kìa. Link tao gửi trong Zalo rồi đấy, lẹ lên không hết mã.",
    "🟢 Ham 6 (Việc khẩn cấp/Có SĐT)": "Alo gọi khẩn cấp cho thầy Tuấn số 0988123456 nhé, thầy đang chờ duyệt tài liệu đồ án ở cổng trường đấy, nhanh lên!",
    "🟢 Ham 7 (Học tập - NLP)": "Ê chiều nay báo cáo bài tập NLP nhóm mình review lại phần TF-IDF với Word2Vec tí nhé. T sợ thầy hỏi xoáy phần code phân loại text đấy.",
    "🟢 Ham 8 (Giải trí - Bóng đá/Game)": "Hôm qua xem trận Arsenal đá hay thực sự. Cuối tuần này rảnh làm vài ván eFootball không? T mới build lại đội hình ngon lắm.",
    "🟢 Ham 9 (Đồ án - Xử lý ảnh)": "File slide báo cáo xử lý ảnh phần nhận diện QR code t gửi qua drive rồi đấy. Mày check xem điều kiện ánh sáng lúc demo trên lớp ổn không nha.",
    "🟢 Ham 10 (Code - Front-end)": "Phần front-end cái app quản lý kho sữa hạt tao đang vướng logic xuất nhập trước sau (FEFO). Mai lên lớp xem hộ tao cái luồng MVC một chút nhé."
}

# ── 2. TẢI MODEL & TÀI NGUYÊN ─────────────────────────────────
@st.cache_resource(show_spinner="Đang nạp hệ thống NLP...")
def load_system():
    tfidf_word   = pickle.load(open(os.path.join(OUTPUT_DIR, "tfidf_vectorizer.pkl"), "rb"))
    tfidf_char   = pickle.load(open(os.path.join(OUTPUT_DIR, "tfidf_char.pkl"), "rb"))
    scaler       = pickle.load(open(os.path.join(OUTPUT_DIR, "manual_scaler.pkl"), "rb"))
    le           = pickle.load(open(os.path.join(OUTPUT_DIR, "label_encoder.pkl"), "rb"))
    best_t       = pickle.load(open(os.path.join(OUTPUT_DIR, "best_threshold.pkl"), "rb"))
    manual_cols  = pickle.load(open(os.path.join(OUTPUT_DIR, "manual_cols.pkl"), "rb"))
    spam_lexicon = pickle.load(open(os.path.join(OUTPUT_DIR, "spam_lexicon.pkl"), "rb"))
    
    # Nạp thêm Word2Vec
    try:
        from gensim.models import Word2Vec
        w2v_model  = Word2Vec.load(os.path.join(OUTPUT_DIR, "word2vec.model"))
        w2v_scaler = pickle.load(open(os.path.join(OUTPUT_DIR, "w2v_scaler.pkl"), "rb"))
    except:
        w2v_model, w2v_scaler = None, None
        
    if os.path.exists(STOPWORDS_PATH):
        with open(STOPWORDS_PATH, 'r', encoding='utf-8') as f:
            stopwords = set([line.strip() for line in f if line.strip()])
    else:
        stopwords = set(["bị", "bởi", "các", "cái", "cho", "có", "của", "đã", "đang", "để", "do", "là", "thì", "mà", "những", "trong", "và", "vào", "với"])
        
    # Đảm bảo return đúng 10 biến
    return tfidf_word, tfidf_char, scaler, le, best_t, manual_cols, spam_lexicon, stopwords, w2v_model, w2v_scaler

@st.cache_resource(show_spinner="Đang nạp Mô hình Học máy...")
def load_ml_model(model_filename):
    return pickle.load(open(os.path.join(OUTPUT_DIR, model_filename), "rb"))

@st.cache_data
def load_results():
    csv_path = os.path.join(OUTPUT_DIR, "buoc4_ket_qua.csv")
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    return None

# Đảm bảo hứng đúng 10 biến ở đây
try:
    tfidf_word, tfidf_char, scaler, le, best_t_loaded, manual_cols, spam_lexicon, VN_STOPWORDS, w2v_model, w2v_scaler = load_system()
except Exception as e:
    st.error(f"❌ Vui lòng chạy file Bước 4 trước. Lỗi: {e}")
    st.stop()



# ── 3. XỬ LÝ NGÔN NGỮ (NLP) ───────────────────────────────────
VN_TEENCODE = {
    r"\bko\b": "không", r"\bkh\b": "không", r"\bk\b": "không", r"\bdc\b": "được", r"\bđc\b": "được", 
    r"\bvs\b": "với", r"\bcx\b": "cũng", r"\bj\b": "gì", r"\bck\b": "chuyển khoản", r"\bstk\b": "số tài khoản", 
    r"\bib\b": "nhắn tin", r"\bsdt\b": "số điện thoại", r"\bkm\b": "khuyến mãi", r"\bvcl\b": "vãi", r"\bvl\b": "vãi" 
}
_PAT_PHONE = re.compile(r"\b(03|05|07|08|09|01[2689])\d{8}\b|\b(1900|1800)\d{4,6}\b")
_PAT_URL = re.compile(r"http[s]?://|www\.|[a-z0-9\-]+\.(vn|com\.vn|com|net|org)/\S+|bit\.ly/|zalo\.me/", re.I)
_PAT_MONEY = re.compile(r"vnđ|vnd|đồng|usd|\b\d+\s*(k|tr|triệu|tỷ|củ|lít)\b|chuyển\s+khoản|số\s+dư", re.I)
_PAT_FREE = re.compile(r"miễn\s+phí|trúng\s+thưởng|khuyến\s+mãi|quà\s+tặng|voucher|sale|vay\s+vốn|lãi\s+suất|giải\s+ngân|cờ\s+bạc|tài\s+xỉu", re.I)

# Thay thế hoàn toàn hàm preprocess_vn cũ bằng đoạn này
VN_SLANG = {
    r"\bko\b": "không", r"\bkh\b": "không", r"\bkhg\b": "không", r"\bkg\b": "không", r"\bk\b": "không",
    r"\bdc\b": "được", r"\bđc\b": "được", r"\bvs\b": "với", r"\bcx\b": "cũng", r"\bj\b": "gì",
    r"\bnt\b": "nhắn tin", r"\bđt\b": "điện thoại", r"\bdt\b": "điện thoại", r"\bib\b": "nhắn tin",
    r"\bsđt\b": "số điện thoại", r"\bsdt\b": "số điện thoại", r"\bqc\b": "quảng cáo", r"\bkm\b": "khuyến mãi",
    r"\bck\b": "chuyển khoản", r"\btk\b": "tài khoản", r"\bstk\b": "số tài khoản", r"\bsp\b": "sản phẩm",
    r"\bh\b": "giờ", r"\bthk\b": "cảm ơn", r"\bntn\b": "như thế nào",
    r"\bhnay\b": "hôm nay", r"\bmai\b": "ngày mai", r"\bvcl\b": "vãi", r"\bvl\b": "vãi"
}

def preprocess_vn(text):
    text = text.lower()
    for p, r in VN_SLANG.items(): text = re.sub(p, r, text)
    text = re.sub(r"http\S+|www\S+", " url ", text)
    text = re.sub(r"\b(0[3|5|7|8|9])+([0-9]{8})\b", " phonenumber ", text)
    
    vn_chars = "a-záàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵđ"
    text = re.sub(f"[^{vn_chars}_\\s]", " ", text)
    
    text = ViTokenizer.tokenize(text) 
    tokens = [t for t in text.split() if t not in VN_STOPWORDS and len(t) > 1]
    return " ".join(tokens)

def compute_lexicon_score(clean_text):
    tokens = clean_text.split()
    return float(np.mean([spam_lexicon.get(t, 0.0) for t in tokens])) if tokens else 0.0

def compute_manual_features(text, clean_text):
    words = text.split()
    all_feats = {
        "num_chars": len(text), "num_words": len(words), "num_sentences": len(re.split(r"[.!?]", text)),
        "has_phone": 1 if _PAT_PHONE.search(text) else 0, "has_url": 1 if _PAT_URL.search(text) else 0,
        "has_money": 1 if _PAT_MONEY.search(text) else 0, "has_free": 1 if _PAT_FREE.search(text) else 0,
        "uppercase_ratio": sum(1 for c in text if c.isupper()) / max(len(text), 1),
        "punct_density": sum(1 for c in text if c in "!?$#@%&*") / max(len(text), 1),
        "digit_ratio": sum(1 for c in text if c.isdigit()) / max(len(text), 1),
        "exclamation_count": text.count("!"), "all_caps_words": sum(1 for w in words if w.isupper() and len(w) > 1),
        "avg_word_len": np.mean([len(w) for w in words]) if words else 0,
        "currency_count": len(re.findall(r"vnđ|vnd|\$|k\b", text.lower())), "lexicon_score": compute_lexicon_score(clean_text)
    }
    return np.array([[all_feats.get(c, 0) for c in manual_cols]], dtype=float)

def predict(text, threshold, clf_model):
    clean  = preprocess_vn(text)
    
    X_word = tfidf_word.transform([clean])
    X_char = tfidf_char.transform([clean])
    X_man  = scaler.transform(compute_manual_features(text, clean))
    
    # Xử lý Word2Vec
    if w2v_model and w2v_scaler:
        tokens = clean.split()
        vecs = [w2v_model.wv[w] for w in tokens if w in w2v_model.wv]
        w2v_doc = np.mean(vecs, axis=0) if len(vecs) > 0 else np.zeros(100)
        X_w2v_sp = sp.csr_matrix(w2v_scaler.transform([w2v_doc]))
        
        # Gộp 4 features
        X_comb = sp.hstack([X_word, X_char, sp.csr_matrix(X_man), X_w2v_sp])
    else:
        # Nếu không nạp được W2V, chạy mặc định
        X_comb = sp.hstack([X_word, X_char, sp.csr_matrix(X_man)])
    
    if hasattr(clf_model, "predict_proba"):
        prob = clf_model.predict_proba(X_comb)[0][list(le.classes_).index("spam")]
    elif hasattr(clf_model, "decision_function"):
        prob = 1 / (1 + np.exp(-clf_model.decision_function(X_comb)[0]))
    else:
        prob = 1.0 if clf_model.predict(X_comb)[0] == list(le.classes_).index("spam") else 0.0
            
    lbl = "spam" if prob >= threshold else "ham"
    return lbl, float(prob), clean


# ── 4. GIAO DIỆN (UI) ─────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🤖 Cấu Hình Mô Hình")
    
    available_models = []
    all_files_in_dir = []
    if os.path.exists(OUTPUT_DIR):
        all_files_in_dir = os.listdir(OUTPUT_DIR)
        for f in all_files_in_dir:
            # Chặn đứng các model cũ liên quan đến PhoBERT không cho load lên
            if f.endswith(".pkl") and "model" in f.lower() and "phobert" not in f.lower():
                available_models.append(f)
                
    if not available_models: available_models = ["best_model.pkl"]
    
    MODEL_NAMES = {
        "best_model.pkl": "🌟 Mô Hình Tốt Nhất (Tự động)",
        "voting_model.pkl": " Voting Ensemble (Tổng hợp)-VE",
        "logistic_reg_model.pkl": " Logistic Regression-LR",
        "logistic_regression_model.pkl": " Logistic Regression-LR",
        "svm_calibrated_model.pkl": " Support Vector Machine-SVM",
        "svm_model.pkl": " Support Vector Machine-SVM",
        "nb_complement_model.pkl": " Naive Bayes (Complement)-CNB",
        "nb_multinomial_model.pkl": " Naive Bayes (Multinomial)-MNB"
    }
        
    selected_model_file = st.selectbox(
        "Chọn Mô Hình Để Phân Tích:", 
        available_models,
        format_func=lambda x: MODEL_NAMES.get(x, x) 
    )
    
    model = load_ml_model(selected_model_file)
    st.success(f"Đang kích hoạt: {type(model).__name__} (TF-IDF)")
    
    st.markdown("---")
    st.markdown("### ⚙️ Ngưỡng Bắt Spam (Threshold)")
    use_optimal = st.toggle("Dùng Ngưỡng Tối Ưu (F2-Score)", value=True)
    threshold   = best_t_loaded if use_optimal else st.slider("Chỉnh tay ngưỡng:", 0.1, 0.9, 0.5, 0.05)
    
    st.markdown("---")
    st.markdown("### 💡 Văn Bản Mẫu")
    selected = st.radio("Thử ngay:", list(SAMPLES.keys()))

    st.markdown("---")
    with st.expander("🛠️ Radar Quét Lỗi (Debug)"):
        st.write(f"Đang đọc thư mục:\n`{OUTPUT_DIR}`")
        st.write("Các file model tìm thấy:")
        st.code("\n".join(available_models) if available_models else "Không tìm thấy file nào!")

st.markdown('<div class="main-title">🛡️ Hệ Thống Nhận Diện Spam Tiếng Việt</div>', unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["🔍 Phân tích tin nhắn", "📊 Đánh giá Mô hình", "📈 Thống kê Dữ liệu"])

# ================= TAB 1 =================
with tab1:
    col1, col2 = st.columns([3, 2])
    with col1:
        st.markdown("#### ✍️ Nhập văn bản")
        user_input = st.text_area("Nội dung tin nhắn / Bài đăng:", value=SAMPLES[selected], height=150, label_visibility="collapsed")
        c1, c2, c3 = st.columns([2, 1, 1])
        with c1: btn = st.button("🔍 Phân Tích Ngay", type="primary", use_container_width=True)
        with c3: batch_mode = st.checkbox("📋 Quét Hàng Loạt")

    with col2:
        st.markdown("#### 📌 Bóc Tách Đặc Trưng")
        if user_input:
            clean_preview = preprocess_vn(user_input)
            lex = compute_lexicon_score(clean_preview)
            c_a, c_b = st.columns(2)
            with c_a:
                st.write(f"{'🔴' if _PAT_PHONE.search(user_input) else '⚪'} Chứa Số điện thoại")
                st.write(f"{'🔴' if _PAT_URL.search(user_input) else '⚪'} Chứa Link / URL")
                st.write(f"{'🔴' if _PAT_MONEY.search(user_input) else '⚪'} Từ khóa Tiền Bạc")
            with c_b:
                st.metric("Tỷ lệ In hoa", f"{sum(1 for c in user_input if c.isupper()) / max(len(user_input), 1):.1%}")
                st.metric("Điểm Từ Vựng", f"{lex:+.2f}", delta="Nghi Spam" if lex > 0 else "Bình thường", delta_color="inverse" if lex > 0 else "normal")

    if btn and user_input.strip():
        lbl, prob, clean = predict(user_input, threshold, model)
        st.markdown("---")
        st.markdown("#### 🎯 KẾT QUẢ DỰ ĐOÁN")
        
        r1, r2 = st.columns([1, 1])
        with r1:
            if lbl == "spam":
                st.markdown(f'<div class="spam-box">🚨 NGUY CƠ SPAM<br><small>Độ tin cậy: {prob*100:.1f}%</small></div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="ham-box">✅ AN TOÀN<br><small>Độ tin cậy: {(1-prob)*100:.1f}%</small></div>', unsafe_allow_html=True)
                
        with r2:
            fig_g, ax_g = plt.subplots(figsize=(5, 2.2))
            ax_g.barh([""], [prob], color="#e74c3c", height=0.4, label="Spam")
            ax_g.barh([""], [1-prob], left=[prob], color="#2ecc71", height=0.4, label="Ham")
            ax_g.axvline(x=threshold, color="black", linestyle="--", lw=1.5, label=f"Ngưỡng ({threshold:.2f})")
            ax_g.set_xlim(0, 1); ax_g.legend(fontsize=8)
            plt.tight_layout(); st.pyplot(fig_g, use_container_width=True); plt.close()
            
        with st.expander("🔎 Xem dữ liệu thô đưa vào Học máy"):
            st.code(clean)

    if batch_mode:
        st.markdown("---")
        st.markdown("#### 📋 Quét Hàng Loạt")
        batch_input = st.text_area("Mỗi dòng 1 văn bản:", height=150, placeholder="Dán danh sách vào đây...")
        if st.button("🔍 Xử lý tất cả") and batch_input.strip():
            lines = [l.strip() for l in batch_input.split("\n") if l.strip()]
            rows, prog = [], st.progress(0)
            for i, line in enumerate(lines):
                lbl_, prob_, _ = predict(line, threshold, model)
                rows.append({"Văn bản": line[:70] + "..." if len(line)>70 else line, "Phân loại": "🚨 SPAM" if lbl_ == "spam" else "✅ BÌNH THƯỜNG", "Xác suất": f"{prob_*100:.1f}%"})
                prog.progress((i+1)/len(lines))
            st.dataframe(pd.DataFrame(rows), use_container_width=True)

# ================= TAB 2 =================
with tab2:
    st.markdown("#### 📊 Bảng Đánh Giá Các Mô Hình")
    res = load_results()
    if res is not None:
        metric_cols = [c for c in ["Accuracy","Precision","Recall","F1-Score","F2-Score(β=2)","AUC"] if c in res.columns]
        fmt = {c: "{:.4f}" for c in metric_cols}
        st.dataframe(res.style.highlight_max(subset=metric_cols, color="#d5f5e3").format(fmt), use_container_width=True)

        # PHỤC HỒI BIỂU ĐỒ SO SÁNH
        fig_r, axes_r = plt.subplots(1, 2, figsize=(14, 5))
        cr = ["#e74c3c","#3498db","#2ecc71","#f39c12","#9b59b6","#1abc9c","#e67e22","#c0392b"]
        xr = np.arange(len(metric_cols)); wr = 0.8 / len(res)
        
        for i, (_, row) in enumerate(res.iterrows()):
            axes_r[0].bar(xr + i*wr - 0.4 + wr/2, [row[m] for m in metric_cols], wr, label=row["Mô hình"][:12], color=cr[i%len(cr)], alpha=0.85)
        
        axes_r[0].set_xticks(xr); axes_r[0].set_xticklabels(metric_cols, rotation=15)
        axes_r[0].set_ylim(0.6, 1.05); axes_r[0].legend(fontsize=6, loc="lower right")
        axes_r[0].axhline(0.95, color="gray", linestyle="--", alpha=0.4)
        axes_r[0].set_title("So sánh các chỉ số (Metrics)", fontweight="bold")

        f2_col = "F2-Score(β=2)" if "F2-Score(β=2)" in res.columns else "F1-Score"
        si = res[f2_col].argsort()[::-1].values
        colors_h = ["#e67e22" if i == 0 else "#3498db" for i in range(len(res))]
        axes_r[1].barh([res.iloc[i]["Mô hình"][:18] for i in si][::-1], [res.iloc[i][f2_col] for i in si][::-1], color=colors_h)
        axes_r[1].set_title(f"Xếp hạng theo {f2_col}", fontweight="bold")
        axes_r[1].axvline(0.95, color="gray", linestyle="--", alpha=0.4)
        plt.tight_layout(); st.pyplot(fig_r); plt.close()
    else:
        st.warning("Chưa có kết quả. Vui lòng chạy file Bước 4.")

# ================= TAB 3 =================
with tab3:
    st.markdown("#### 📈 Phân Tích Dữ Liệu Huấn Luyện")
    data_path = os.path.join(BASE_DIR, "vietnamese_data.csv")
    if os.path.exists(data_path):
        df_d = pd.read_csv(data_path)
        if "spam" in df_d.columns:
            df_d["label"] = df_d["spam"].map({0: "ham", 1: "spam"})
            c1, c2, c3 = st.columns(3)
            c1.metric("Tổng số mẫu", len(df_d))
            c2.metric("Mẫu Bình thường (Ham)", (df_d.label=="ham").sum())
            c3.metric("Mẫu Rác (Spam)", (df_d.label=="spam").sum())

            # PHỤC HỒI BIỂU ĐỒ THỐNG KÊ
            fig_d, axes_d = plt.subplots(1, 2, figsize=(12, 5))
            
            # Biểu đồ Tròn
            lc2 = df_d.label.value_counts()
            axes_d[0].pie(lc2.values, labels=lc2.index, colors=["#2ecc71","#e74c3c"], autopct="%1.1f%%", startangle=90)
            axes_d[0].set_title("Phân bổ Nhãn dữ liệu", fontweight="bold")

            # Binary Features
            feat_list  = ["has_phone","has_url","has_money","has_free"]
            fname_list = ["Có SĐT","Có Link","Từ khóa Tiền","Từ khóa Free"]
            
            ham_feats = [np.mean([1 if re.search(p, str(t).lower()) else 0 for t in df_d[df_d.label=="ham"]["text"]])*100 for p in [_PAT_PHONE, _PAT_URL, _PAT_MONEY, _PAT_FREE]]
            spam_feats = [np.mean([1 if re.search(p, str(t).lower()) else 0 for t in df_d[df_d.label=="spam"]["text"]])*100 for p in [_PAT_PHONE, _PAT_URL, _PAT_MONEY, _PAT_FREE]]

            x_f = np.arange(4); w_f = 0.35
            axes_d[1].bar(x_f-w_f/2, ham_feats, w_f, label="Ham", color="#2ecc71")
            axes_d[1].bar(x_f+w_f/2, spam_feats, w_f, label="Spam", color="#e74c3c")
            axes_d[1].set_xticks(x_f); axes_d[1].set_xticklabels(fname_list, rotation=15)
            axes_d[1].set_title("Tỷ lệ xuất hiện các Đặc trưng (%)", fontweight="bold")
            axes_d[1].legend()

            plt.tight_layout(); st.pyplot(fig_d); plt.close()
    else:
        st.warning("Cần file `vietnamese_data.csv` cùng thư mục để hiển thị phần này.")
