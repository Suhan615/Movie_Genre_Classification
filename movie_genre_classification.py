from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

def load_data(file_path):
    plots = []
    genres = []
    with open(file_path, encoding="utf-8") as file:
        for line in file:
            parts = line.split(":::")
            if len(parts) >= 4:
                genres.append(parts[2].strip())
                plots.append(parts[3].strip())
    return plots, genres

# Load training data
X_train, y_train = load_data("train_data.txt")

# Load testing data (with labels)
X_test, y_test = load_data("test_data_solution.txt")

# Convert text to numbers
vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train model
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Predict and evaluate
predictions = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, predictions)

print("Model Accuracy:", accuracy)

# Test custom input
sample_movie = ["A group of friends go on a dangerous adventure to save the world"]
sample_vec = vectorizer.transform(sample_movie)
print("Predicted Genre:", model.predict(sample_vec)[0])
