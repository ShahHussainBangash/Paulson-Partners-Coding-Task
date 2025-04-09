# Paulson-Partners-Coding-Task
Developed a simplified AI assistant that detects basic emotions—happiness, sadness, and anger—from text using a pre-trained model (e.g., Hugging Face Transformers). Integrated it with a conversational agent that adapts its tone to match the user's emotion for more natural interaction.


Shah Hussain Bangash NLP & Machine Developer

1.Important Libraries Used in This Task:- Panda / Matplotlib / Seaborn / wordcloud / re / String / NLTK / Stopwords / NTLK_Tokenize / Spacy / Hugging Face Tokenizers  - AutoTokenizer / Sklearn / Numpy / 

2.Used Dataset:- Tweet Emotions.CSV       70% Data for Tranning | 30% for Testing
Features: Empty / Anger / Happiness / Sadness / Enthusiasasm / Neutral / Worry / Superise / Love / Fun etc.

3.Machine Learning Algorithms : - 
1)TF-IDF - Logistic Regression 
2)TF-IDF - Naive Bayes 
3)TF-IDF - Naive Bayes 
4)TF-IDF Support Vector Machine (SVM)

Natural Language Algorithms :- Hugging-Face Transformers
5)Distilbert-base-uncased-emotion
6)Tone-Adaptive Conversational Agent

Step 1: Import Libraries and Load Dataset
pandas is used for data manipulation.
The CSV file contains tweet text and associated emotion labels.

Step 2: Explore Dataset
Understand data types and identify missing values.
Cleaning data is crucial before model training.

Step 3: Visualize Emotion Labels
Helps detect imbalanced datasets which may affect model performance.

Step 4: Preprocess Text Data
Removes noise like punctuation, links, HTML, digits, etc.
Essential for clean input to NLP models.

Step 5: Tokenization and Stop Words Removal
Removes common words that do not contribute to sentiment (e.g., is, the, and).

Step 6: Convert Text to Vectors (TF-IDF)
Machine learning models require numerical input.
TF-IDF gives importance to meaningful words.

Step 7: Encode Labels (Emotions)
Converts emotion categories (like happy, sad) into numerical values.

Step 8: Split Dataset into Train Data 70% / Test Data 30%
Splitting ensures we can test model performance on unseen data.

Step 9: Train a Machine Learning Model
Trains a Naive Bayes classifier and evaluates performance using accuracy and classification report.

Step 10: Build Conversational AI Component
Takes user input and returns predicted emotion.
Integrates all preprocessing and model inference into one function.
