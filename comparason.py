import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import os

nltk.download("punkt")
nltk.download("stopwords")


# Function to preprocess and tokenize a document
def preprocess_and_tokenize(document):
    # Read the document
    with open(document, "r") as file:
        text = file.read()

    # Tokenize the text
    tokens = word_tokenize(text)

    # Remove punctuation and convert to lowercase
    tokens = [word.lower() for word in tokens if word.isalpha()]

    # Remove stopwords
    stop_words = set(stopwords.words("english"))
    tokens = [word for word in tokens if word not in stop_words]

    # Stem the words
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]

    return " ".join(tokens)


# List of documents
documents = [
    "Dokument 1.txt",
    "Dokument 2.txt",
    "Dokument 3.txt",
    "Dokument 4.txt",
    "Dokument 5.txt",
]

# Preprocess and tokenize the documents
tokenized_documents = [preprocess_and_tokenize(doc) for doc in documents]

# Calculate TF-IDF vectors
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(tokenized_documents)

# Calculate cosine similarities between the documents
similarities = cosine_similarity(tfidf_matrix)

# Create a table with the similarities
data = []
for i in range(5):
    row = [f"Document{i + 1}.txt"]
    for j in range(5):
        similarity_percentage = similarities[i][j] * 100
        row.append(f"{similarity_percentage:.2f}%")
    data.append(row)

# Create a pandas DataFrame
columns = ["Document"] + [f"Document{i + 1}" for i in range(5)]
df = pd.DataFrame(data, columns=columns)

# Display the table
print(df)
