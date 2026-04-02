import pandas as pd
import numpy as np
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

def engineer_features(df: pd.DataFrame, text_column: str = "text") -> pd.DataFrame:
    """
    Engineers robust features tailored exclusively to the insights gained from the EDA.
    
    Modifications based on EDA:
    - Included log transformations for wildly skewed length distributions.
    - Added high-value semantic markers (links, phones, questions).
    - DROPPED `num_caps` and `caps_ratio` since the dataset is heavily pre-lowercased (mean=1).
    - ADDED critical removal of Domain Leakage words (Enron specific terms) from text.
    """
    df = df.copy()
    
    # 1. Base Structural Features
    df["num_chars"] = df[text_column].str.len()
    
    # We use basic string splitting for speed if tokenizer isn't crucial for pure lengths,
    # but since NLTK is standard for words/sentences, we'll keep it.
    df["num_words"] = df[text_column].apply(lambda x: len(word_tokenize(str(x))))
    df["num_sentences"] = df[text_column].apply(lambda x: len(nltk.sent_tokenize(str(x))))
    
    # 2. Skewness Correction (Mandatory based on EDA density plots)
    df["log_num_chars"] = np.log1p(df["num_chars"])
    df["log_num_words"] = np.log1p(df["num_words"])
    df["log_num_sentences"] = np.log1p(df["num_sentences"])
    
    # 3. Retained Engineered Meta-features
    df["num_digits"] = df[text_column].apply(lambda x: sum(1 for c in str(x) if c.isdigit()))
    df["num_special"] = df[text_column].apply(lambda x: sum(1 for c in str(x) if c in string.punctuation))
    df["has_link"] = df[text_column].apply(lambda x: 1 if re.search(r"http|www|\.com", str(x), flags=re.IGNORECASE) else 0)
    df["has_phone"] = df[text_column].apply(lambda x: 1 if re.search(r"\d{10}", str(x)) else 0)
    df["num_exclamations"] = df[text_column].apply(lambda x: str(x).count("!"))
    df["num_questions"] = df[text_column].apply(lambda x: str(x).count("?"))
    
    # Note: `num_caps` intentionally omitted as it has exactly zero variance.

    # 4. Critical Token Cleaning (Defeating Overfitting & Leakage)
    nltk.download("stopwords", quiet=True)
    stop_words_base = set(stopwords.words("english"))
    
    # These were the massive offenders driving up model accuracy artificially
    engine_leakage_words = {
        "enron", "ect", "vince", "kaminski", "hou", 
        "subject", "cc", "j", "pm", "com", "www", "http"
    }
    
    final_stopwords = stop_words_base.union(engine_leakage_words)
    
    def apply_cleaning(text):
        tokens = word_tokenize(str(text))
        # Keep only alphanumeric tokens (stripping raw punctuation strings like '_' and '.')
        filtered = [
            word.lower() for word in tokens 
            if word.isalnum() and word.lower() not in final_stopwords
        ]
        return " ".join(filtered)
        
    df["text_clean"] = df[text_column].apply(apply_cleaning)
    
    return df
