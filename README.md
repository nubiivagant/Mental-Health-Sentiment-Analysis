# 🧠 Mental Health Sentiment Detection from Social Media Posts

This project focuses on **Mental Health Sentiment Detection** using textual data from social media platforms such as **Twitter** and **Reddit**.  
The primary goal is to automatically classify posts expressing mental health–related discussions into a variety of **Anxiety**, **Bipolar**, **Depression** etc. sentiments.  
The system helps identify emotional tone patterns in public communication related to topics like **depression**, **anxiety**, and **mental well-being**.

A comparative analysis was performed using two Machine Learning models — **Support Vector Machine (SVM)** and **Logistic Regression (LR)** — both implemented on two different datasets using identical preprocessing and feature extraction techniques.

---

## 💡 Why is it important?

Mental health has become a critical global concern, with social media emerging as an open platform for users to express their emotions.  
Analyzing these sentiments can assist in understanding public well-being, early detection of distress signals, and forming preventive mental health strategies.

This project applies **Natural Language Processing (NLP)** to automate the process of detecting sentiments from social posts, enabling large-scale, real-time monitoring of mental health trends.

## 📊 Dataset Overview

### **Dataset 1 – Mental Health Twitter Dataset (Kaggle)**
- Contains tweets related to **stress, anxiety, depression, and well-being**.  
- Each tweet includes text and a sentiment label (*positive*, *negative*, *neutral*).  
- Balanced dataset suitable for TF-IDF vectorization and ML classification.  

### **Dataset 2 – Mental Health Reddit Dataset (Kaggle)**
- Contains Reddit posts and comments from **mental health–related subreddits**.  
- Includes manually annotated sentiment labels.  
- Provides varied sentence structures and context-rich discussions.  

### **Preprocessing and Cleaning**
- Removal of URLs, hashtags, user mentions, emojis, and punctuation.
- Lowercasing, tokenization, lemmatization, and stopword removal.  
- Vectorization using **TF-IDF** (max features = 5000).  
- Sentiment labels encoded for supervised learning.

