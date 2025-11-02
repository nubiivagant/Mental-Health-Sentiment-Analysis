"""
Streamlit App for Mental Health Sentiment Detection
Loads trained models and provides interactive predictions
"""

import streamlit as st
import pandas as pd
import numpy as np
import re
import string
import nltk
from nltk.corpus import stopwords
import joblib
import json

# Page configuration
st.set_page_config(
    page_title="Mental Health Sentiment Detection",
    page_icon="🧠",
    layout="wide"
)

# Initialize NLTK stopwords
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    stop_words = set(stopwords.words('english'))
except LookupError:
    nltk.download('stopwords', quiet=True)
    stop_words = set(stopwords.words('english'))


@st.cache_resource
def load_models():
    """Load all trained models and artifacts"""
    models = {}
    try:
        # Load TF-IDF
        models['tfidf'] = joblib.load('models/tfidf.pkl')
        
        # Load models
        models['lr'] = joblib.load('models/lr_model.pkl')
        models['svm'] = joblib.load('models/svm_model.pkl')
        
    # Load label mapping
        with open('models/label_list.json', 'r') as f:
            models['label_mapping'] = json.load(f)
        
    # LSTM artifacts removed; only LR and SVM expected in models/
        
        return models, None
    except FileNotFoundError as e:
        return None, f"Model file not found: {e}. Please train the models first using the Jupyter notebook."
    except Exception as e:
        return None, f"Error loading models: {e}"


def clean_text(text):
    """
    Comprehensive text cleaning function (same as in notebook)
    - Remove URLs
    - Remove mentions (@user) and hashtags
    - Remove punctuation and numbers
    - Convert to lowercase
    - Remove stopwords
    - Remove extra whitespace
    """
    if pd.isna(text) or text == "":
        return ""
    
    text = str(text)
    
    # Remove URLs
    text = re.sub(r'http\S+|www.\S+', '', text)
    
    # Remove mentions and hashtags
    text = re.sub(r'@\w+|#\w+', '', text)
    
    # Remove punctuation and numbers
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove stopwords
    words = text.split()
    words = [w for w in words if w not in stop_words and len(w) > 2]
    
    # Join and clean whitespace
    text = ' '.join(words)
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def predict_lr(text, models):
    """Predict using Logistic Regression"""
    cleaned = clean_text(text)
    if not cleaned:
        return "Empty text after cleaning", 0.0
    
    # Transform using TF-IDF
    text_tfidf = models['tfidf'].transform([cleaned])
    
    # Predict
    prediction = models['lr'].predict(text_tfidf)[0]
    probabilities = models['lr'].predict_proba(text_tfidf)[0]
    confidence = float(np.max(probabilities))
    
    # Map to label (handle both string and int keys)
    pred_key = str(int(prediction))
    if pred_key in models['label_mapping']:
        label = models['label_mapping'][pred_key]
    else:
        # Fallback: try integer key
        label = models['label_mapping'].get(int(prediction), f"Class {prediction}")
    
    return label, confidence


def predict_svm(text, models):
    """Predict using SVM"""
    cleaned = clean_text(text)
    if not cleaned:
        return "Empty text after cleaning", 0.0
    
    # Transform using TF-IDF
    text_tfidf = models['tfidf'].transform([cleaned])
    
    # Predict
    prediction = models['svm'].predict(text_tfidf)[0]

    # Try to get class probabilities; not all SVMs implement predict_proba
    if hasattr(models['svm'], 'predict_proba'):
        probabilities = models['svm'].predict_proba(text_tfidf)[0]
    else:
        # Fall back to decision_function and convert to pseudo-probabilities via softmax
        if hasattr(models['svm'], 'decision_function'):
            dec = models['svm'].decision_function(text_tfidf)
            # decision_function can return shape (n_classes,) or (n_samples, n_classes)
            dec_vals = np.asarray(dec)
            if dec_vals.ndim == 1:
                dec_vals = dec_vals.reshape(1, -1)
            # apply softmax across classes
            def _softmax(x):
                e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
                return e_x / e_x.sum(axis=1, keepdims=True)
            probabilities = _softmax(dec_vals)[0]
        else:
            # As a last resort, return one-hot for predicted class
            probs = np.zeros(len(models['label_mapping']))
            try:
                cls_index = list(models['label_mapping'].keys()).index(str(int(prediction)))
            except Exception:
                cls_index = 0
            probs[cls_index] = 1.0
            probabilities = probs

    confidence = float(np.max(probabilities))
    
    # Map to label (handle both string and int keys)
    pred_key = str(int(prediction))
    if pred_key in models['label_mapping']:
        label = models['label_mapping'][pred_key]
    else:
        # Fallback: try integer key
        label = models['label_mapping'].get(int(prediction), f"Class {prediction}")
    
    return label, confidence


#def predict_lstm(text, models):
    # LSTM removed from project; function kept as placeholder in case needed later
#    raise NotImplementedError("LSTM model removed. Use Logistic Regression or SVM.")


# Main App
def main():
    st.title("Mental Health Sentiment Detection")
    st.markdown("Analyze sentiment in social media posts")
    
    # Load models
    models, error = load_models()
    
    if error:
        st.error(error)
        st.info("Please run the Jupyter notebook first to train and save the models.")
        return
    
    st.success("All models loaded successfully!")
    
    # Sidebar with model info
    with st.sidebar:
        st.header("Model Information")
        st.write("This app uses two models:")
        st.write("- Logistic Regression")
        st.write("- SVM")
        
        st.header("Labels")
        if models['label_mapping']:
            for idx, label in models['label_mapping'].items():
                st.markdown(f"- {label}")
    
    # Main input area
    st.header("Enter Text for Analysis")
    
    # Text input
    user_text = st.text_area(
        "Enter a tweet or Reddit post:",
        height=150
    )
    
    # Predict button
    if st.button("Analyze Sentiment", type="primary"):
        if not user_text or user_text.strip() == "":
            st.warning("Please enter some text to analyze.")
        else:
            # Show original text
            st.markdown("---")
            st.subheader("Original Text")
            st.write(user_text)
            
            # Show cleaned text
            cleaned_text = clean_text(user_text)
            with st.expander("View Cleaned Text"):
                st.write(cleaned_text if cleaned_text else "Text became empty after cleaning")
            
            # Get predictions from all models
            st.markdown("---")
            st.subheader("Model Predictions")
            
            # Create columns for predictions (2 models)
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### Logistic Regression")
                lr_label, lr_conf = predict_lr(user_text, models)
                st.metric("Prediction", lr_label)
                st.metric("Confidence", f"{lr_conf:.2%}")
                # Progress bar for confidence
                st.progress(lr_conf)
            
            with col2:
                st.markdown("### SVM")
                svm_label, svm_conf = predict_svm(user_text, models)
                st.metric("Prediction", svm_label)
                st.metric("Confidence", f"{svm_conf:.2%}")
                st.progress(svm_conf)
            
            # Comparison section
            st.markdown("---")
            st.subheader("Prediction Comparison")
            
            # Create comparison DataFrame
            comparison_df = pd.DataFrame({
                'Model': ['Logistic Regression', 'SVM'],
                'Prediction': [lr_label, svm_label],
                'Confidence': [lr_conf, svm_conf]
            })
            
            st.dataframe(comparison_df, use_container_width=True, hide_index=True)
            
            # Agreement check
            predictions = [lr_label, svm_label]
            unique_predictions = set(predictions)
            if len(unique_predictions) == 1:
                st.success(f"All models agree: {lr_label}")
            else:
                st.warning(f"Models have different predictions: {', '.join(unique_predictions)}")
    
    # Example section
    with st.expander("Example Inputs"):
        examples = [
            "I've been struggling with anxiety and depression for months now. It's really hard to cope.",
            "Feeling great today! Just finished a therapy session and feeling more positive.",
            "Not sure what I'm feeling. Some days are good, some days are bad.",
            "I need help but don't know where to turn. Mental health services are hard to access."
        ]
        
        for i, example in enumerate(examples, 1):
            if st.button(f"Use Example {i}", key=f"example_{i}"):
                st.session_state['example_text'] = example
                st.rerun()
    
    # Check if example was selected
    if 'example_text' in st.session_state:
        user_text = st.session_state['example_text']
        del st.session_state['example_text']
        # Automatically trigger prediction
        # Note: Streamlit doesn't support auto-triggering buttons, so we'll just show the text
        st.text_area("Enter a tweet or Reddit post:", value=user_text, height=150, key="example_input")


if __name__ == "__main__":
    main()
