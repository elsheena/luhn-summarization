import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from collections import Counter

nltk.download('punkt')
nltk.download('stopwords')

# Luhn Summarization
def luhn_summarization(text, num_sentences=3):
    # Tokenize sentences
    sentences = sent_tokenize(text)
    # Tokenize words and remove stopwords
    stop_words = set(stopwords.words("english"))
    words = word_tokenize(text.lower())
    significant_words = [word for word in words if word.isalnum() and word not in stop_words]

    # Word frequencies
    freq = Counter(significant_words)
    threshold = sum(freq.values()) / len(freq)

    # Sentence scoring
    sentence_scores = {}
    for sentence in sentences:
        sentence_words = word_tokenize(sentence.lower())
        significant = [word for word in sentence_words if word in freq and freq[word] >= threshold]
        if significant:
            sentence_scores[sentence] = len(significant)**2 / len(sentence_words)

    # Sort by score and return top sentences
    top_sentences = sorted(sentence_scores, key=sentence_scores.get, reverse=True)[:num_sentences]
    return " ".join(top_sentences)

# Test the implementation
if __name__ == "__main__":
    with open("test_file.txt", "r") as file:
        blog_post = file.read()

    summary = luhn_summarization(blog_post)
    print("Summary:\n", summary)
