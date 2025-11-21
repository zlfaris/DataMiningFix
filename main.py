import streamlit as st
import joblib
import re

st.set_page_config(
    page_title="Analisis Sentimen Tweet",
    page_icon="💬",
    layout="centered"
)

st.title("💬 Analisis Sentimen Tweet")
st.write("Masukkan teks tweet kemudian klik tombol analisis untuk melihat sentimennya.")

@st.cache_resource
def load_models():
    try:
        model_bnb = joblib.load("model_bernoulli_nb.pkl")
        model_svm = joblib.load("model_linear_svm.pkl")
        model_ensemble = joblib.load("model_ensemble_voting.pkl")
        vectorizer = joblib.load("vectorizer_tfidf.pkl")
        return model_bnb, model_svm, model_ensemble, vectorizer
    except Exception as e:
        st.error(f"❌ Gagal memuat model: {e}")
        return None, None, None, None

model_bnb, model_svm, model_ensemble, vectorizer = load_models()

if not all([model_bnb, model_svm, model_ensemble, vectorizer]):
    st.stop()

def clean_text(x):
    x = str(x).lower()
    x = re.sub(r"http\S+", "", x)
    x = re.sub(r"@\w+", "", x)
    x = re.sub(r"[^a-z ]+", " ", x)
    x = re.sub(r"\s+", " ", x).strip()
    return x

st.subheader("✍️ Masukkan Tweet")

example_texts = [
    "Aku suka banget layanan mereka!",
    "Parah sih ini, bikin kesel banget.",
    "Lumayan, tidak buruk juga.",
    "Sangat memuaskan!",
    "Kecewa berat sama pelayanan ini."
]

selected_example = st.selectbox(
    "Pilih contoh:",
    ["-- Ketik manual --"] + example_texts
)

default_text = "" if selected_example == "-- Ketik manual --" else selected_example

text = st.text_area("Teks Tweet:", value=default_text, height=120)

if st.button("🔍 Analisis Sentimen", type="primary"):

    if text.strip() == "":
        st.warning("Masukkan teks terlebih dahulu.")
        st.stop()

    try:
        cleaned = clean_text(text)
        vec = vectorizer.transform([cleaned])

        pred = model_ensemble.predict(vec)[0]
        prob = model_ensemble.predict_proba(vec)[0]
        classes = model_ensemble.classes_

        st.subheader("🎯 Hasil Analisis")
        st.success(f"Sentimen: **{pred.upper()}**")

        st.write("Probabilitas:")
        for c, p in zip(classes, prob):
            st.write(f"- **{c}** : {p*100:.1f}%")

        st.caption(f"Preprocessing: `{cleaned}`")

    except Exception as e:
        st.error(f"❌ Terjadi error: {e}")
