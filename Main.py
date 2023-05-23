import tkinter as tk
from tkinter import messagebox
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load the dataset from a CSV file
data = pd.read_csv(r'C:\Users\ahmed\OneDrive - Arab Academy for Science and Technology\Desktop\mo.csv')

# Define a list of stop words
stop_words = set(stopwords.words('english'))

# Initialize the WordNet lemmatizer
lemmatizer = WordNetLemmatizer()

# Function to preprocess user input
def preprocess_input(user_input):
    # Tokenize the input text
    user_input = str(user_input)  # Convert input_text to string if it's an integer
    tokens = word_tokenize(user_input.lower())

    # Remove stop words and perform lemmatization
    preprocessed_tokens = [
        lemmatizer.lemmatize(token) for token in tokens if token not in stop_words
    ]

    return " ".join(preprocessed_tokens)

# Preprocess the "Churn" column
data['Churn'] = data['Churn'].map({'Yes': 1, 'No': 0})

# Preprocess the text columns for prediction
text_columns = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService',
                'MultipleLines', 'InternetService', 'OnlineSecurity', 'Contract',
                'PaymentMethod']

for column in text_columns:
    data[column] = data[column].apply(preprocess_input)

# Concatenate all text columns into a single input feature
data['user_input'] = data[text_columns].apply(lambda x: ' '.join(x), axis=1)

# Split the dataset into input features and target variable
X = data['user_input']
y = data['Churn']

# Create a TF-IDF vectorizer
vectorizer = TfidfVectorizer()

# Vectorize the input features
X_vectorized = vectorizer.fit_transform(X)

# Train a logistic regression model
model = LogisticRegression()
model.fit(X_vectorized, y)

# Function to predict churn based on user input
def predict_churn(user_input):
    preprocessed_input = preprocess_input(user_input)
    input_vectorized = vectorizer.transform([preprocessed_input])
    churn_prediction = model.predict(input_vectorized)[0]
    if churn_prediction == 1:
        return 'Churn: Yes'
    else:
        return 'Churn: No'

# Create main window
window = tk.Tk()
window.title("Churn Prediction Chatbot")
window.geometry("600x400") # Set window size
window.config(bg="#F0F0F0") # Set window background color

# Create title label
title_label = tk.Label(window, text="Churn Prediction Chatbot", font=("Arial", 20), fg="#0000FF", bg="#F0F0F0")
title_label.grid(row=0, column=0, columnspan=2, padx=10, pady=10)

# Create labels and entry fields for each text column
text_columns = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService',
                'MultipleLines', 'InternetService', 'OnlineSecurity', 'Contract',
                'PaymentMethod']

labels = {} # Dictionary to store label widgets
entries = {} # Dictionary to store entry widgets

for i, column in enumerate(text_columns):
    # Create label widget
    labels[column] = tk.Label(window, text=column.capitalize() + ":", font=("Arial", 12), fg="#000000", bg="#F0F0F0")
    labels[column].grid(row=i+1, column=0, padx=10, pady=5, sticky="e")

    # Create entry widget
    entries[column] = tk.Entry(window, width=30)
    entries[column].grid(row=i+1, column=1, padx=10, pady=5)

# Function to get user input from entry fields
def get_user_input():
    user_input = ""
    for column in text_columns:
        user_input += entries[column].get() + " "
    return user_input.strip()

# Function to clear user input from entry fields
def clear_user_input():
    for column in text_columns:
        entries[column].delete(0, tk.END)

# Function to validate user input
def validate_user_input():
    valid = True
    for column in text_columns:
        if not entries[column].get():
            valid = False
            break
    return valid

# Function to predict churn based on user input
def predict_churn_gui():
    try:
        if validate_user_input():
            user_input = get_user_input()
            churn_prediction = predict_churn(user_input)
            messagebox.showinfo("Churn Prediction", churn_prediction)
        else:
            messagebox.showwarning("Warning", "Please enter values for all fields.")
    except Exception as e:
        messagebox.showerror("Error", "Something went wrong. Please try again.")
        print(e)

# Create predict button
predict_button = tk.Button(window, text="Predict", font=("Arial", 12), fg="#FFFFFF", bg="#0000FF", command=predict_churn_gui)
predict_button.grid(row=12, column=1, padx=10, pady=10)

# Create clear button
clear_button = tk.Button(window, text="Clear", font=("Arial", 12), fg="#FFFFFF", bg="#0000FF", command=clear_user_input)
clear_button.grid(row=12, column=2, padx=10, pady=10)

# Start GUI event loop
window.mainloop()