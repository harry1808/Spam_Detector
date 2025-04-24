# Spam_Detector
This project is about detecting the spam in YouTube comments using Machine Learning Algorithms.
This project demonstrates the application of various **machine learning algorithms** to detect spam in YouTube video comments. By using text preprocessing and multiple classification techniques, we build a robust spam classifier evaluated on standard metrics.
## About the Project
Spam comments on YouTube degrade the quality of conversations and user experience. This project aims to build a supervised machine learning pipeline to classify comments as **spam** or **ham (not spam)** using TF-IDF vectorization and several ML algorithms.

---

##Technologies Used 
- **Python 3.9+**
- **Google Colab** for experimentation
- **scikit-learn** for machine learning
- **imbalanced-learn** for handling class imbalance
- **pandas**, **NumPy**, **matplotlib**, **seaborn** for data manipulation and visualization

## Installation
1. Clone the repository:
- git clone https://github.com/harry1808/Spam_Detector.git 
- cd Spam_Detector

2 **.(Optional)** Create a Virtual environment:
bash
- python -m venv venv
- source venv/bin/activate  # On Windows use venv\Scripts\activate


3. Install dependencies:
 bash

 - pip install -r requirements.txt
  
## **Usage** 
1. Open the Jupytor notebook:
bash
jupyter notebook Spam_Detection_of_YouTube_Video_Comments.ipynb

2. Or run it directly in **Google Colab**

   - Upload the .ipynb file to Google Colab
   - Follow the code cells in order

## **Models Implemented**

- Logistic Regression
- Multinomial Naive Bayes
- Support Vector Machine (SVM)
- Random Forest
- Decision Tree
- K-Nearest Neighbors
- Multi-layer Perceptron (MLP)
- Voting Classifier (Ensemble)

## **Evalutation Metrics**

Each model is evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC Score
Visualizations like ROC curves and confusion matrices are used for comparative analysis.

## **Project Results**
- The project highlights how different classifiers perform on text data with class imbalance.
- Ensamble method and TF-IDF representation provide improved classification performance.
- See the notebook for detailed plots and metric values.

## **📂 Dataset**
- The Dataset consist of You Tube video comments labeled as spam or ham.
- I used the YouTube Video Comments online dataset in Kaggle:
link:

     https://www.kaggle.com/datasets/lakshmi25npathi/images/data

## **🙌 Acknowledgements**
- Google Colab
- scikit-learn
- imbalanced-learn
- YouTube community datasets (if publically sourced)
