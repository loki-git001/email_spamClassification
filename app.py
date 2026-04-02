import streamlit as st
import pandas as pd
import joblib
import nltk

from feature_engineering import engineer_features

# Pre-download required NLTK data silently to prevent issues on first run
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)

# Page Configuration
st.set_page_config(
    page_title="Email Spam Classifier", 
    page_icon="📧", 
    layout="centered"
)

@st.cache_resource
def load_model():
    """Loads the SVM pipeline and caches it so it doesn't reload on every button click."""
    return joblib.load('models/svm_pipeline.joblib')

st.title("📧 Email Spam Classifier")
st.markdown("Analyze incoming emails to detect whether they are **Spam** or **Safe (Ham)** using a robust Support Vector Machine (SVM) algorithm.")

model = load_model()

email_input = st.text_area("Paste the email content below:", height=250)

if st.button("Classify Email", type="primary"):
    if not email_input.strip():
        st.warning("Please enter some text to classify.")
    else:
        with st.spinner("Analyzing semantics and structure..."):
            # 1. Structure raw input into a DataFrame
            df_raw = pd.DataFrame({"text": [email_input]})
            
            # 2. Extract robust structural features (log transformations, specific markers, and text cleaning)
            df_processed = engineer_features(df_raw, text_column="text")
            
            # 3. Predict classification and probabilities using the Pipeline
            prediction = model.predict(df_processed)[0]
            probabilities = model.predict_proba(df_processed)[0]
            
            # Label map: 0 = Ham, 1 = Spam
            is_spam = bool(prediction == 1)
            
            # Display Results
            st.divider()
            
            if is_spam:
                st.error("🚨 **Classification: SPAM**")
                spam_prob = probabilities[1]
                st.markdown(f"**Confidence Rate (Spam):** {spam_prob * 100:.1f}%")
                st.progress(float(spam_prob))
            else:
                st.success("✅ **Classification: SAFE (HAM)**")
                ham_prob = probabilities[0]
                st.markdown(f"**Confidence Rate (Safe):** {ham_prob * 100:.1f}%")
                st.progress(float(ham_prob))
