import numpy as np
import pandas as pd
import math

from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


data = pd.read_csv("cipher.csv")


def entropy(text):
    counts = Counter(text)
    total = len(text)

    return -sum(
        (count / total) * math.log2(count / total)
        for count in counts.values()
    )


def index_of_coincidence(text):
    counts = Counter(text)
    n = len(text)

    if n <= 1:
        return 0

    return sum(
        count * (count - 1)
        for count in counts.values()
    ) / (n * (n - 1))


def repeated_ratio(text):
    if len(text) == 0:
        return 0

    return (len(text) - len(set(text))) / len(text)


def vowel_ratio(text):
    if len(text) == 0:
        return 0

    vowels = "AEIOU"
    return sum(c in vowels for c in text) / len(text)


def consonant_ratio(text):
    if len(text) == 0:
        return 0

    vowels = "AEIOU"
    return sum(c.isalpha() and c not in vowels for c in text) / len(text)


def bigram_repetition(text):
    if len(text) < 2:
        return 0

    bigrams = [
        text[i:i+2]
        for i in range(len(text) - 1)
    ]

    return 1 - (len(set(bigrams)) / len(bigrams))


def trigram_repetition(text):
    if len(text) < 3:
        return 0

    trigrams = [
        text[i:i+3]
        for i in range(len(text) - 2)
    ]

    return 1 - (len(set(trigrams)) / len(trigrams))


data["length"] = data["ciphertext"].str.len()

data["unique_ratio"] = data["ciphertext"].apply(
    lambda x: len(set(x)) / len(x)
)

data["repeat_ratio"] = data["ciphertext"].apply(
    repeated_ratio
)

data["entropy"] = data["ciphertext"].apply(
    entropy
)

data["IC"] = data["ciphertext"].apply(
    index_of_coincidence
)

data["vowel_ratio"] = data["ciphertext"].apply(
    vowel_ratio
)

data["consonant_ratio"] = data["ciphertext"].apply(
    consonant_ratio
)

data["bigram_repeat"] = data["ciphertext"].apply(
    bigram_repetition
)

data["trigram_repeat"] = data["ciphertext"].apply(
    trigram_repetition
)


features = [
    "length",
    "unique_ratio",
    "repeat_ratio",
    "entropy",
    "IC",
    "vowel_ratio",
    "consonant_ratio",
    "bigram_repeat",
    "trigram_repeat"
]


X = data[features]
y = data["algorithm"]


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)


prediction = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, prediction))

print("\nClassification Report:")
print(classification_report(y_test, prediction))


print("\nFeature Importance:")

importance = pd.DataFrame({
    "feature": features,
    "importance": model.feature_importances_
})

print(
    importance.sort_values(
        "importance",
        ascending=False
    )
)


print("\nPredictions:")

probabilities = model.predict_proba(X_test)

for i in range(len(X_test)):
    print(
        "Actual:",
        y_test.iloc[i],
        "| Predicted:",
        prediction[i],
        "| Probability:",
        round(max(probabilities[i]) * 100, 2),
        "%"
    )


def extract_features(text):
    return pd.DataFrame([{
        "length": len(text),
        "unique_ratio": len(set(text)) / len(text),
        "repeat_ratio": repeated_ratio(text),
        "entropy": entropy(text),
        "IC": index_of_coincidence(text),
        "vowel_ratio": vowel_ratio(text),
        "consonant_ratio": consonant_ratio(text),
        "bigram_repeat": bigram_repetition(text),
        "trigram_repeat": trigram_repetition(text)
    }])


ciphertext = input("Enter ciphertext: ").upper()

features = extract_features(ciphertext)

prediction = model.predict(features)[0]

probabilities = model.predict_proba(features)[0]

print("\nPredicted Algorithm:", prediction)

print("\nProbabilities:")

for algorithm, probability in zip(model.classes_, probabilities):
    print(
        algorithm,
        ":",
        round(probability * 100, 2),
        "%"
    )