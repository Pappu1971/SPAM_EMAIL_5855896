import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, GRU, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import joblib

# Hide Streamlit and TF warnings for cleaner UI
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Configure modern page layout
st.set_page_config(
    page_title="AI Spam Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
<style>
.main-header {
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    padding: 2rem;
    border-radius: 15px;
    color: white;
    text-align: center;
    margin-bottom: 2rem;
    box-shadow: 0 4px 20px rgba(0,0,0,0.1);
}

.metric-card {
    background: white;
    padding: 1.5rem;
    border-radius: 12px;
    border-left: 4px solid #667eea;
    box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    margin-bottom: 1rem;
}

.prediction-box {
    background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    padding: 2rem;
    border-radius: 15px;
    border: 2px solid #e0e0e0;
    margin: 1rem 0;
}

.success-alert {
    background: linear-gradient(90deg, #56ab2f 0%, #a8e6cf 100%);
    color: white;
    padding: 1rem;
    border-radius: 10px;
    margin: 1rem 0;
}

.warning-alert {
    background: linear-gradient(90deg, #ff6b6b 0%, #ffa726 100%);
    color: white;
    padding: 1rem;
    border-radius: 10px;
    margin: 1rem 0;
}
</style>
""", unsafe_allow_html=True)

# ---- 1. MODERN HEADER ----
st.markdown("""
<div class="main-header">
    <h1>🛡️ AI-Powered Spam Detection System</h1>
    <p style="font-size: 1.2rem; margin-top: 1rem;">
        Advanced LSTM/GRU Neural Network with Self-Learning Capabilities
    </p>
</div>
""", unsafe_allow_html=True)

# Feature highlights in columns
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("""
    <div class="metric-card">
        <h3>🧠 Self-Learning</h3>
        <p>Model adapts with your feedback in real-time</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="metric-card">
        <h3>📊 Live Analytics</h3>
        <p>Real-time performance metrics and visualizations</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="metric-card">
        <h3>🎯 High Accuracy</h3>
        <p>Advanced bidirectional LSTM & GRU architecture</p>
    </div>
    """, unsafe_allow_html=True)

with st.expander("ℹ️ Project Explanation & Logic (click to expand)", expanded=False):
    st.markdown("""
    **Project Structure:**
    1. **Dataset:** Emails split into subject, message, and spam/ham label.
    2. **Preprocessing:** 
       - Texts are tokenized and padded for neural network input.
    3. **Model:**
       - Uses embedding, bidirectional LSTM & GRU for advanced sequence learning.
       - Output: Probability of 'spam' or 'ham'.
    4. **Self-Learning:**
       - If the prediction is wrong, your correction is instantly added and the model updates.
    5. **Evaluation:**
       - Shows confusion matrix, accuracy, F1, and classification report.
       
    **Key Libraries:**
    - `TensorFlow / Keras`: Deep learning model.
    - `scikit-learn`: Evaluation metrics, train/test split.
    - `pandas`: Data handling.
    - `matplotlib / seaborn`: Visualization.
    - `streamlit`: Interactive web UI.
    """)

# ---- 2. LOAD DATA AND INITIALIZE (runs only once per session) ----
@st.cache_resource
def load_data_and_model():
    df = pd.read_csv("spam_email_format.csv")
    df['label'] = df['label'].map({'spam': 1, 'ham': 0})
    df['text'] = df['subject'].fillna('') + ' ' + df['message'].fillna('')
    tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")
    tokenizer.fit_on_texts(df['text'])
    maxlen = 50
    X = pad_sequences(tokenizer.texts_to_sequences(df['text']), maxlen=maxlen, padding='post')
    y = df['label'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
    
    if os.path.exists("spam_lstm_model.keras"):
        model = load_model("spam_lstm_model.keras")
    else:
        model = tf.keras.Sequential([
            Embedding(5000, 64, input_length=maxlen),
            Bidirectional(LSTM(32, return_sequences=True)),
            Dropout(0.3),
            Bidirectional(GRU(16)),
            Dense(32, activation='relu'),
            Dropout(0.3),
            Dense(1, activation='sigmoid')
        ])
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.fit(X_train, y_train, epochs=3, batch_size=32, validation_split=0.1)
        model.save("spam_lstm_model.keras")
    return df, tokenizer, model, X_train, y_train, X_test, y_test, maxlen

df, tokenizer, model, X_train, y_train, X_test, y_test, maxlen = load_data_and_model()

# ---- 3. FUNCTIONS FOR PREDICTION & LEARNING ----
def clean_input(text):
    if not isinstance(text, str):
        return ""
    return text.strip() if text.strip() else ""

def predict_email(subject, message):
    subject = clean_input(subject)
    message = clean_input(message)
    if not subject and not message:
        return "Invalid input: Both subject and message are empty.", 0.0
    text = subject + " " + message
    seq = tokenizer.texts_to_sequences([text])
    pad = pad_sequences(seq, maxlen=maxlen, padding='post')
    pred = model.predict(pad, verbose=0)[0][0]
    return ('spam' if pred > 0.5 else 'ham'), float(pred)

def self_learn(subject, message, label):
    global X_train, y_train, model
    text = clean_input(subject) + " " + clean_input(message)
    seq = tokenizer.texts_to_sequences([text])
    pad = pad_sequences(seq, maxlen=maxlen, padding='post')
    X_train = np.concatenate([X_train, pad])
    y_train = np.concatenate([y_train, [1 if label == 'spam' else 0]])
    model.fit(X_train, y_train, epochs=1, batch_size=32, validation_split=0.1, verbose=0)
    model.save("spam_lstm_model.keras")

# ---- 4. MODERN EMAIL PREDICTION UI ----
st.markdown("## 🔍 Test Email Classification")

# Sidebar for quick examples
with st.sidebar:
    st.markdown("### 📝 Quick Test Examples")
    if st.button("💰 Lottery Spam Example"):
        st.session_state.example_subject = "CONGRATULATIONS! You've Won $1,000,000!"
        st.session_state.example_message = "Click here to claim your prize now! Limited time offer!"
    
    if st.button("📧 Normal Email Example"):
        st.session_state.example_subject = "Meeting reminder for tomorrow"
        st.session_state.example_message = "Hi, just wanted to remind you about our meeting tomorrow at 2 PM."

# Main prediction interface
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("""
    <div class="prediction-box">
    """, unsafe_allow_html=True)
    
    with st.form("prediction_form"):
        subject = st.text_input(
            "📬 Email Subject", 
            value=st.session_state.get('example_subject', ''),
            placeholder="Enter email subject here..."
        )
        message = st.text_area(
            "📄 Email Message", 
            value=st.session_state.get('example_message', ''),
            placeholder="Enter email content here...",
            height=150
        )
        submitted = st.form_submit_button("🔍 Analyze Email", use_container_width=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("### 📊 Model Stats")
    st.metric("Model Type", "Bi-LSTM + GRU")
    st.metric("Vocab Size", "5,000 words")
    st.metric("Sequence Length", "50 tokens")

if submitted:
    with st.spinner("🤖 Analyzing email..."):
        pred, score = predict_email(subject, message)
    
    # Modern prediction display
    if pred == 'spam':
        st.markdown(f"""
        <div class="warning-alert">
            <h2>🚨 SPAM DETECTED</h2>
            <p style="font-size: 1.2rem;">Confidence: {score:.1%}</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="success-alert">
            <h2>✅ LEGITIMATE EMAIL</h2>
            <p style="font-size: 1.2rem;">Confidence: {(1-score):.1%}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Feedback section
    st.markdown("### 🎯 Help Improve the Model")
    feedback_col1, feedback_col2 = st.columns(2)
    
    with feedback_col1:
        feedback = st.radio("Was this prediction correct?", ("✅ Yes, correct", "❌ No, incorrect"))
    
    if feedback == "❌ No, incorrect":
        with feedback_col2:
            correct_label = st.radio("What should it be?", ("🚨 Spam", "✅ Ham"))
            correct_label = 'spam' if correct_label == "🚨 Spam" else 'ham'
        
        if st.button("🔄 Update Model", use_container_width=True):
            with st.spinner("🧠 Learning from your feedback..."):
                self_learn(subject, message, correct_label)
            st.markdown("""
            <div class="success-alert">
                🎉 Model updated successfully! The AI learned from your feedback.
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")

# ---- 5. MODERN PERFORMANCE DASHBOARD ----
st.markdown("## 📈 Performance Analytics Dashboard")

# Performance metrics with modern cards
perf_col1, perf_col2, perf_col3, perf_col4 = st.columns(4)

with st.spinner("🔄 Calculating live metrics..."):
    y_pred = (model.predict(X_test) > 0.5).astype(int).flatten()
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    precision = (y_pred & y_test).sum() / y_pred.sum() if y_pred.sum() > 0 else 0
    recall = (y_pred & y_test).sum() / y_test.sum() if y_test.sum() > 0 else 0

with perf_col1:
    st.metric("🎯 Accuracy", f"{acc:.1%}", f"{(acc-0.95)*100:+.1f}%")

with perf_col2:
    st.metric("⚖️ F1 Score", f"{f1:.3f}", f"{(f1-0.90)*100:+.1f}%")

with perf_col3:
    st.metric("🎪 Precision", f"{precision:.1%}")

with perf_col4:
    st.metric("📡 Recall", f"{recall:.1%}")

# Confusion Matrix and Classification Report in columns
chart_col1, chart_col2 = st.columns([1, 1])

with chart_col1:
    st.markdown("### 🔥 Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='viridis', 
                xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'], ax=ax)
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title("Model Prediction Matrix", fontsize=14, fontweight='bold')
    st.pyplot(fig)

with chart_col2:
    st.markdown("### 📊 Detailed Classification Report")
    report = classification_report(y_test, y_pred, target_names=['Ham','Spam'], digits=3)
    st.code(report, language='text')
    
    # Distribution chart
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    unique, counts = np.unique(y_test, return_counts=True)
    labels = ['Ham', 'Spam']
    colors = ['#56ab2f', '#ff6b6b']
    ax2.pie(counts, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
    ax2.set_title("Test Set Distribution", fontsize=14, fontweight='bold')
    st.pyplot(fig2)

# Expandable sections with modern styling
with st.expander("📋 Sample Data Preview", expanded=False):
    st.markdown("### Dataset Structure")
    st.dataframe(df.head(10), use_container_width=True)

with st.expander("🛠️ Technical Stack & Architecture", expanded=False):
    tech_col1, tech_col2 = st.columns(2)
    
    with tech_col1:
        st.markdown("""
        **🧠 Deep Learning Stack:**
        - TensorFlow 2.x + Keras
        - Bidirectional LSTM layers
        - GRU (Gated Recurrent Units)
        - Dropout regularization
        - Adam optimizer
        """)
    
    with tech_col2:
        st.markdown("""
        **📊 Data & Visualization:**
        - pandas for data manipulation
        - scikit-learn for metrics
        - matplotlib + seaborn for plots
        - Streamlit for web interface
        - Real-time model updates
        """)

# Modern footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 2rem; background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); 
            border-radius: 10px; color: white; margin-top: 2rem;">
    <h4>🚀 Advanced AI Spam Detection System</h4>
    <p>© 2025 | PAPPU SINGHA 5855896 | Powered by TensorFlow & Streamlit</p>
</div>
""", unsafe_allow_html=True)
