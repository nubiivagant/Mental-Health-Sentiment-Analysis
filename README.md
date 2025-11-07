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

## 🛠️ Model Implementation

<img width="600" height="600" alt="image" src="https://github.com/user-attachments/assets/7d9087fe-f289-4a8f-be44-527515837312" />

This is the System Architecture of the project.

The workflow consists of four primary steps, as outlined in the system architecture diagram:

1.  **Data Cleaning and Preprocessing**
    * **Data Cleaning:** Remove noise, unnecessary tokens, symbols, mentions, URLs, punctuation, and emojis from the text data.
    * **Text Normalization:** Performed using stopword removal, lemmatization, and lowercasing. Tokenization is also done.
    * **Vectorization:** Apply **TF-IDF Vectorizer** with a vocabulary size of **5000** to convert the text into numerical feature representations.
    * **Label Encoding:** Sentiment labels are transformed into numerical classes for training the models.

2.  **Model Training**
    * **Logistic Regression (LR):** A fast linear model efficient for linearly separable text data. Class weights were balanced, and regularization strength (C) was optimized.
    * **Support Vector Machine (SVM):** Utilizes a linear kernel SVM to achieve well-separated sentiment classification. Confidence scoring is facilitated by setting `probability=True`.

3.  **Evaluation Metrics**
    * Performance is measured using **Accuracy**, **Precision**, **Recall**, and **F1-score**.
    * A **Confusion Matrix** visualization is used to compare performance across sentiment categories.

4.  **Comparison Analysis**
    * Both models were tested on the same datasets to ensure consistency and generality.

## 🚀 Steps to Run the Code

1.  **Clone the repository**
    ```bash
    git clone [https://github.com/nubiivagant/Mental-Health-Sentiment-Analysis.git](https://github.com/nubiivagant/Mental-Health-Sentiment-Analysis.git)
    cd Mental-Health-Sentiment-Analysis
    ```
2.  **Run the Jupyter notebook**
    ```bash
    jupyter notebook "ML Project.ipynb"
    ```
    This step handles the **Data preprocessing**, **Model training**, and **evaluation**. The trained models are saved in the `models/` folder.
3.  **Run the Streamlit app**
    ```bash
    streamlit run app.py
    ```
4.  **Utilize the Interface**
    * Enter a mental health-related statement into the user interface.
    * The app will present the **Sentiment prediction** (Anxiety, Depression, Bipolar) and the **confidence score** of each model.

  ## 📊 Results and Analysis

### Model Evaluation and Comparison

The Support Vector Machine (SVM) slightly outperformed Logistic Regression (LR) in both primary metrics.

| Model | Accuracy Score | F1 Score | Interpretation |
| :--- | :--- | :--- | :--- |
| **Logistic Regression (LR)** | 0.7499 | 0.7435 | The baseline model reveals impressive linear separability and equilibrium of metrics. |
| **Support Vector Machine (SVM)** | 0.7562 | 0.7517 | The model, using a linear kernel, is slightly better in both measures, supporting the handling of subtle emotional language. |


The comparable performance values, with SVM showing a small **+0.6% gain** in accuracy, support the reliability of the robust preprocessing and the effectiveness of the **TF-IDF representation** in successfully extracting the most significant sentiment-bearing words.

### Confusion Matrix Comparison

The confusion matrices for both classifiers show similar results across the seven sentiment categories, indicating strong generalization across the datasets.
<img width="1061" height="345" alt="Screenshot 2025-11-02 215534" src="https://github.com/user-attachments/assets/ba54b13e-c71d-4d04-bb74-488a28ec3ce8" />


### Key Findings Summary

* **SVM Performance:** SVM slightly outperformed Logistic Regression on every metric, achieving an accuracy of **0.7562** and an F1 score of **0.7517**.
* **LR Speed:** Logistic Regression was noted to be a bit faster.
* **Feature Extraction:** The **TF-IDF vectorization** was highly effective in identifying sentiment-bearing words without needing deep embeddings, proving that basic machine learning algorithms with strong preprocessing can effectively detect sentiment dimensions.
* **Generalization:** Both models demonstrated strong generalization capabilities with consistent accuracy across the combined Twitter and Reddit datasets.

---

## 🎓 What Was Learned from This Project

This project provided valuable insights into the end-to-end workflow of applying **Machine Learning** and **Natural Language Processing (NLP)** to real-world textual data.  
Through model design, evaluation, and deployment, several important lessons and technical skills were gained.

### **1️⃣ Understanding Text Data Processing**
- Learned the importance of **data cleaning and preprocessing** — including tokenization, lemmatization, and stopword removal — to handle noisy and unstructured social media data.  
- Realized how proper preprocessing significantly improves model accuracy and reduces misclassification.  

### **2️⃣ Feature Extraction and Representation**
- Explored **TF-IDF vectorization** to convert text into meaningful numerical features.  
- Understood how term weighting captures contextual relevance without requiring deep embeddings.  
- Learned that even traditional text representations like TF-IDF can perform competitively when preprocessing is thorough.

### **3️⃣ Model Training and Comparison**
- Observed how **Logistic Regression** serves as an efficient, interpretable baseline for sentiment tasks.  
- Learned that **Support Vector Machines (SVM)** offer slightly better performance in complex, non-linear text relationships.  
- Developed skills in **hyperparameter tuning**, **model evaluation**, and interpreting **F1-scores** and **confusion matrices**.  

### **4️⃣ Evaluation and Visualization**
- Understood how to analyze model effectiveness using **Accuracy**, **Precision**, **Recall**, and **F1-score**.  
- Gained the ability to interpret **confusion matrices** and extract insights into class-wise performance.  
- Learned the importance of visualizations (bar charts, heatmaps) in summarizing model outcomes effectively.

### **5️⃣ Practical Application and Deployment**
- Built a fully functional **Streamlit web interface** for real-time sentiment detection.  
- Learned the process of **saving, loading, and integrating trained models** into a live application.  
- Realized the value of user interactivity in ML systems, allowing non-technical users to benefit from model predictions.

### **6️⃣ Broader Insights**
- Demonstrated that **traditional ML algorithms**, when combined with proper preprocessing, can yield robust sentiment analysis results without the need for deep learning.  
- Highlighted how **machine learning can contribute to mental health monitoring**, offering scalable tools for understanding emotional well-being trends online.  
- Reinforced that performance consistency across multiple datasets validates a model’s generalization capability.

---

**In summary:**  
This project strengthened understanding of how to combine data preprocessing, feature engineering, and machine learning models to address a socially significant problem.  
It emphasized that even with simple architectures, **structured workflows and thoughtful evaluation** can deliver impactful, interpretable, and deployable AI solutions.
