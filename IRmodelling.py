import nltk
from nltk.tokenize import word_tokenize
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from wordcloud import WordCloud
import matplotlib.pyplot as plt

#  pip install nltk=3.8.1 textblob scikit-learn wordcloud matplotlib

#Reading from text file
# with open('doc1.txt', 'r', encoding='utf-8') as f:
#     doc1 = f.read()

# with open('doc2.txt', 'r', encoding='utf-8') as f:
#     doc2 = f.read()

nltk.download("punkt")

doc1 = "I love the new phone I bought. The camera quality is amazing and the battery lasts long."
doc2 = "The new laptop is terrible. It crashes often and the battery life is disappointing."

#Tokenization
token1 = word_tokenize(doc1)
token2 = word_tokenize(doc2)

print("doc1 Tokens : ",token1)
print("doc2 Tokens : ",token2)

#POS Tagging

nltk.download("averaged_perceptron_tagger")
pos1 = nltk.pos_tag(token1)
pos2=  nltk.pos_tag(token2)
print("POS tagging doc1 : ",pos1)
print("POS tagging doc2 : ",pos2)

# Sentiment Analysis
sent1 = TextBlob(doc1).sentiment
sent2 = TextBlob(doc2).sentiment
print("doc1 Sent : ",sent1)
print("doc2 sent : ",sent2)

#cosine similarity
vectorizer = TfidfVectorizer()
tfidf = vectorizer.fit_transform([doc1,doc2])
cosinesim = cosine_similarity(tfidf[0:1], tfidf[1:2])
print("Cosine similarity : ",cosinesim[0][0])

#wordcloud
wcloud1 = WordCloud(background_color = 'white').generate(doc1)
plt.figure(figsize=(8,4))
plt.title("WordCloud for doc1 ")
plt.imshow(wcloud1,interpolation="bilinear")
plt.show()

wcloud2 = WordCloud(background_color = 'white').generate(doc2)
plt.figure(figsize=(8,4))
plt.title("WordCloud for doc2 ")
plt.imshow(wcloud2,interpolation="bilinear")
plt.show()





