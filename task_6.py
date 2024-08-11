import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt

# Generate sample time series data
dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='D')
np.random.seed(0)
values = np.cumsum(np.random.randn(len(dates))) + np.sin(np.arange(len(dates)) * 2 * np.pi / 365) * 10
ts = pd.Series(values, index=dates)

# Split into train and test sets
train = ts[:'2023-06-30']
test = ts['2023-07-01':]

# Fit ARIMA model
model = ARIMA(train, order=(1, 1, 1))  # (p,d,q) order
fitted_model = model.fit()

# Forecast
forecast = fitted_model.forecast(steps=len(test))

# Plot results
plt.figure(figsize=(12, 6))
plt.plot(train, label='Training Data')
plt.plot(test, label='Actual Test Data')
plt.plot(forecast, label='Forecast', color='red')
plt.legend()
plt.title('Time Series Forecast')
plt.show()

# Print forecast accuracy
from sklearn.metrics import mean_squared_error
rmse = np.sqrt(mean_squared_error(test, forecast))
print(f'Root Mean Squared Error: {rmse}')

#movie reviews

import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import movie_reviews
import random

# Download necessary NLTK data
nltk.download('movie_reviews')
nltk.download('vader_lexicon')

# Prepare the data
documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]

random.shuffle(documents)

# Function to get features from words
def get_features(words):
    return dict([(word, True) for word in words])

# Prepare feature sets
featuresets = [(get_features(d), c) for (d, c) in documents]
train_set, test_set = featuresets[100:], featuresets[:100]

# Train a Naive Bayes classifier
classifier = nltk.NaiveBayesClassifier.train(train_set)

# Test the classifier
accuracy = nltk.classify.accuracy(classifier, test_set)
print(f"Naive Bayes Accuracy: {accuracy:.2f}")

# Use VADER for sentiment intensity analysis
sia = SentimentIntensityAnalyzer()

# Example sentences
sentences = [
    "This movie was absolutely fantastic!",
    "The acting was terrible and the plot made no sense.",
    "It was an okay film, nothing special."
]

for sentence in sentences:
    print(f"\nSentence: {sentence}")
    sentiment_scores = sia.polarity_scores(sentence)
    print(f"Sentiment Scores: {sentiment_scores}")


#k- means clustering result

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# Generate sample data
n_samples = 300
n_features = 2
n_clusters = 3
random_state = 42

X, y = make_blobs(n_samples=n_samples, n_features=n_features, centers=n_clusters, 
                  random_state=random_state)

# Perform K-means clustering
kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
kmeans.fit(X)

# Plot the results
plt.figure(figsize=(10, 7))
scatter = plt.scatter(X[:, 0], X[:, 1], c=kmeans.labels_, cmap='viridis')
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.75, marker='x')
plt.title('K-Means Clustering Results')
plt.colorbar(scatter)
plt.show()

# Print cluster centers
print("Cluster Centers:")
for i, center in enumerate(centers):
    print(f"Cluster {i+1}: {center}")

# Predict cluster for new data points
new_points = np.array([[0, 0], [4, 4]])
predictions = kmeans.predict(new_points)
print("\nPredictions for new points:")
for point, cluster in zip(new_points, predictions):
    print(f"Point {point} belongs to cluster {cluster+1}")
