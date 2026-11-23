
import re
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# WordNet lemmatiser
lemmatizer = WordNetLemmatizer()

# Check that WordNet is installed
lemmatizer.lemmatize("trials")

# Keep important medical negation words
STOP_WORDS = set(ENGLISH_STOP_WORDS) - {
    "not", "no", "nor", "without"
}


def clean_text(text):

    if text is None:
        return ""

    text = str(text).lower()

    # Remove punctuation and special characters
    text = re.sub(r"[^a-z\s]", " ", text)

    # Tokenisation
    words = re.findall(r"\b[a-z]{2,}\b", text)

    cleaned_words = []

    for word in words:

        if word not in STOP_WORDS:
            word = lemmatizer.lemmatize(word)
            cleaned_words.append(word)

    return " ".join(cleaned_words)
