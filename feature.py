# feature.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import SMOTE
import streamlit as st

@st.cache_data
def prepare_features(df):
    # 1. Data Validation
    if not {'steming_data', 'Sentiment'}.issubset(df.columns):
        raise ValueError("DataFrame must contain 'steming_data' and 'Sentiment' columns")
    
    # 2. Identical Split (matches original code)
    X = df['steming_data']
    y = df['Sentiment']
    
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,  # Fixed random state
        stratify=y
    )
    
    # 3. TF-IDF with identical parameters
    tfidf = TfidfVectorizer(
    max_features=7000,
    ngram_range=(1, 2),
    stop_words='english',
    random_state=42  # Tambahkan ini
)
    
    # 4. Identical transformation process
    X_train_tfidf = tfidf.fit_transform(X_train_raw)
    X_test_tfidf = tfidf.transform(X_test_raw)
    
    # 5. SMOTE with identical parameters
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train_tfidf, y_train)
    
    # Debug information
    debug_info = {
        'train_samples': len(X_train_raw),
        'test_samples': len(X_test_raw),
        'smote_samples': len(y_train_smote),
        'class_distribution': {
            'original': y.value_counts().to_dict(),
            'train': pd.Series(y_train).value_counts().to_dict(),
            'test': pd.Series(y_test).value_counts().to_dict(),
            'smote': pd.Series(y_train_smote).value_counts().to_dict()
        }
    }
    
    return X_train_smote, X_test_tfidf, y_train_smote, y_test, tfidf, debug_info

@st.cache_data
def get_feature_names(tfidf):
    """Identical feature names extraction"""
    return tfidf.get_feature_names_out()

@st.cache_data
def transform_new_text(tfidf, text):
    """Identical text transformation"""
    return tfidf.transform([text])