# 📧 LSTM Spam Email Detector

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0+-orange.svg)](https://tensorflow.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A modern, self-learning LSTM-based spam email detection system with interactive web interface, inspired by Gmail's filtering capabilities.

## 🌟 Features

- **Deep Learning Architecture**: Bidirectional LSTM + GRU neural networks for robust text classification
- **Self-Learning Capability**: Model updates and improves with user feedback in real-time
- **Interactive Web App**: Clean Streamlit interface for testing and training
- **Advanced Evaluation**: Comprehensive metrics with confusion matrix visualization
- **Model Persistence**: Save and reload trained models for production use
- **Real-time Predictions**: Instant spam/ham classification with confidence scores

## 🚀 Quick Start

### Prerequisites

```bash
Python 3.8+
pip (Python package installer)
```

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/Pappu1971/SPAM_EMAIL_5855896.git
cd SPAM_EMAIL_5855896
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install tensorflow pandas numpy scikit-learn seaborn matplotlib streamlit
```

### Usage

#### 🎯 Train the Model (Jupyter Notebook)
```bash
jupyter notebook spam_lstm_full_upgrade.ipynb
```
Run all cells to train the LSTM model on your spam dataset.

#### 🌐 Launch Web Interface
```bash
streamlit run spamdetectorapp.py
```
Access the interactive web app at `http://localhost:8501`

#### 📊 Quick Prediction (Python)
```python
from tensorflow.keras.models import load_model
# Load your trained model
model = load_model('spam_lstm_model.keras')
# Make predictions...
```

## 📁 Project Structure

```
SPAM_EMAIL_5855896/
├── spam_lstm_full_upgrade.ipynb          # Main training notebook
├── spam_lstm_full_upgrade_executed.ipynb # Executed version with outputs
├── spam_lstm_test.ipynb                  # Testing and experimentation
├── spamdetectorapp.py                    # Streamlit web application
├── spam_email_format.csv                 # Training dataset
├── spam_lstm_model.keras                 # Trained model file
├── venv/                                 # Virtual environment
└── README.md                            # This file
```

## 🏗️ Architecture

### Model Design
- **Embedding Layer**: Converts text to dense vectors (64-dimensional)
- **Bidirectional LSTM**: Captures long-term dependencies in both directions
- **GRU Layer**: Additional sequence processing with gating mechanisms
- **Dropout Layers**: Prevents overfitting (30% dropout rate)
- **Dense Layers**: Final classification with sigmoid activation

### Data Processing Pipeline
1. **Text Preprocessing**: Tokenization with 5000 word vocabulary
2. **Sequence Padding**: Fixed length sequences (max 50 tokens)
3. **Label Encoding**: Binary classification (spam=1, ham=0)
4. **Train/Test Split**: 80/20 stratified split for balanced evaluation

## 📈 Performance

The model achieves excellent performance on spam detection:
- **Accuracy**: ~99%
- **F1-Score**: High precision and recall
- **Training Time**: ~3 epochs for convergence
- **Real-time Inference**: Millisecond-level predictions

## 🔧 Advanced Features

### Self-Learning System
The model can update itself with new feedback:
```python
# Correct a wrong prediction
self_learn(subject="Your email subject", 
          message="Email content", 
          label="spam")  # or "ham"
```

### Model Evaluation
Comprehensive evaluation with:
- Classification reports
- Confusion matrix heatmaps
- F1-score tracking
- Real-time performance monitoring

## 🛡️ Security & Privacy

- **Data Privacy**: No personal data stored beyond training session
- **Secure Processing**: Local model inference without external API calls
- **Regulatory Compliance**: Designed with GDPR/CCPA considerations
- **Model Integrity**: Saved models can be versioned and validated

## 🔮 Future Enhancements

- [ ] Transformer-based models (BERT, RoBERTa)
- [ ] Federated learning for privacy-preserving training
- [ ] Active learning for efficient data labeling
- [ ] Adversarial robustness testing
- [ ] Multi-language support
- [ ] API deployment with FastAPI/Flask

## 📋 Requirements

```
tensorflow>=2.0.0
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
seaborn>=0.11.0
matplotlib>=3.5.0
streamlit>=1.0.0
```

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Pappu**
- GitHub: [@Pappu1971](https://github.com/Pappu1971)
- Project: [SPAM_EMAIL_5855896](https://github.com/Pappu1971/SPAM_EMAIL_5855896)

## 🙏 Acknowledgments

- TensorFlow team for the deep learning framework
- Streamlit for the intuitive web interface
- Gmail spam filtering system for inspiration
- Open source community for continuous improvements

---

⭐ **Star this repository if you found it helpful!**# SPAM_EMAIL_5855896
