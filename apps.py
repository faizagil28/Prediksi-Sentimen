import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import SMOTE
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, classification_report

# Set up Streamlit page
st.set_page_config(page_title="OVO Sentiment Analysis", layout="wide")
st.title("🔍 Analisis Sentimen Ulasan Pengguna OVO")

# Sidebar Instructions
st.sidebar.subheader("Persyaratan File")
st.sidebar.markdown("""
**Format File yang Didukung:**
- CSV (`.csv`)
- Excel (`.xlsx`, `.xls`)

**Struktur File yang Direkomendasikan:**
- Kolom yang wajib: `Sentiment`, `steming_data`
- Contoh isi:
  | Sentiment | steming_data |
  |-----------|---------------|
  | positif   | aplikasi bagus dan cepat |
""")

# Contoh Data
if st.sidebar.checkbox("Tampilkan Contoh Data Valid"):
    contoh = pd.DataFrame({
        'Sentiment': ['positif', 'negatif', 'netral'],
        'steming_data': ['aplikasi bagus', 'layanan lambat', 'cukup baik']
    })
    st.sidebar.dataframe(contoh)

# Fungsi untuk memproses data dan melatih model
def process_data(df):
    X = df['steming_data']
    y = df['Sentiment']
    
    # Split data
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # TF-IDF Vectorizer
    tfidf = TfidfVectorizer(
        max_features=7000,
        ngram_range=(1, 2),
        stop_words='english'
    )
    X_train_tfidf = tfidf.fit_transform(X_train_raw)
    X_test_tfidf = tfidf.transform(X_test_raw)
    
    # SMOTE Oversampling
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train_tfidf, y_train)
    
    # Train model
    nb_model = MultinomialNB(alpha=0.5)
    nb_model.fit(X_train_smote, y_train_smote)
    
    # Make predictions
    y_pred = nb_model.predict(X_test_tfidf)
    
    return nb_model, X_test_tfidf, y_test, y_pred, tfidf

# Main App
def main():
    uploaded_file = st.file_uploader("Pilih file dataset (CSV, XLSX, atau XLS)", type=["csv", "xlsx", "xls"])
    
    if uploaded_file is not None:
        try:
            # Read file
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file, sep=';', encoding='utf-8-sig')
            else:
                df = pd.read_excel(uploaded_file)
            
            if {'Sentiment', 'steming_data'}.issubset(df.columns):
                st.success("✅ Data berhasil dimuat!")
                st.dataframe(df.head())
                
                # Training section
                st.subheader("🔧 Proses Training Model")
                with st.spinner("Sedang memproses data..."):
                    model, X_test, y_test, y_pred, tfidf = process_data(df)
                    
                    # Display evaluation results
                    st.subheader("📊 Hasil Evaluasi Model")
                    
                    # Confusion Matrix
                    st.subheader("Confusion Matrix")
                    cm = confusion_matrix(y_test, y_pred)
                    fig, ax = plt.subplots(figsize=(8, 6))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                                xticklabels=model.classes_, yticklabels=model.classes_)
                    plt.xlabel('Prediksi')
                    plt.ylabel('Aktual')
                    st.pyplot(fig)
                    
                    # Classification Report
                    st.subheader("Classification Report")
                    report = classification_report(y_test, y_pred, output_dict=True)
                    report_df = pd.DataFrame(report).transpose()
                    
                    # Format the report for better display
                    formatted_report = report_df.style.format({
                        'precision': '{:.2f}',
                        'recall': '{:.2f}',
                        'f1-score': '{:.2f}',
                        'support': '{:.0f}'
                    })
                    
                    st.dataframe(formatted_report)
                    
                    # Main Metrics
                    st.subheader("Metrik Utama")
                    cols = st.columns(4)
                    cols[0].metric("Accuracy", f"{report['accuracy']:.2f}")
                    cols[1].metric("Precision", f"{report['weighted avg']['precision']:.2f}")
                    cols[2].metric("Recall", f"{report['weighted avg']['recall']:.2f}")
                    cols[3].metric("F1-Score", f"{report['weighted avg']['f1-score']:.2f}")
                    
                    st.success("✅ Model berhasil dilatih dengan Naive Bayes Multinomial!")
                    
                    # Prediction
                    st.subheader("🔮 Prediksi Sentimen dari Input Teks")
                    user_input = st.text_area("Masukkan teks ulasan:")
                    if st.button("Prediksi"):
                        if user_input.strip():
                            vector_input = tfidf.transform([user_input])
                            prediction = model.predict(vector_input)[0]
                            st.success(f"Prediksi Sentimen: **{prediction.upper()}**")
                        else:
                            st.warning("Mohon masukkan teks terlebih dahulu.")
            else:
                st.error("Dataset harus memiliki kolom 'Sentiment' dan 'steming_data'.")
        except Exception as e:
            st.error(f"Terjadi kesalahan: {str(e)}")
if __name__ == "__main__":
    main()