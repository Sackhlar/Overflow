import re
import string
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from gensim.corpora import Dictionary
from gensim.models import LdaModel
from bs4 import BeautifulSoup
import nltk
import pickle
import pandas as pd

nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('stopwords')

# Initialisation globale du lemmatiseur, du modèle LDA et du dictionnaire
lemmatizer = None
lda_model = None
dictionary = None

def init():
    global lemmatizer, lda_model, dictionary

    # Télécharger les ressources nécessaires
    nltk.download('averaged_perceptron_tagger')
    nltk.download('wordnet')
    nltk.download('stopwords')

    # Initialiser le lemmatiseur de NLTK
    lemmatizer = WordNetLemmatizer()



    # Charger le dictionnaire
    with open('C:/Users/matth/Documents/stack/dictionary.pkl', 'rb') as f:
        dictionary = pickle.load(f)

# Initialiser le script
init()


# Fonction pour le prétraitement du texte
def preprocess_text(text):
    # Convertir le texte en minuscules
    text = text.lower()

    # Supprimer les URL
    text = re.sub(r'https?://\S+|www\.\S+', '', text)

    # Supprimer les balises HTML
    text = BeautifulSoup(text, "lxml").get_text()
    
    # Ajouter des espaces entre les mots collés
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    
    # Supprimer la ponctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Lemmatisation avec NLTK et suppression des stopwords
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text)
    text = " ".join([lemmatizer.lemmatize(token) for token in tokens if token not in stop_words])
    
    return text

# Supposez que df est votre dataframe initial
df = pd.read_csv("C:/Users/matth/python/QueryResults.csv")

# Réduisez la taille de votre dataframe à 5000 lignes
df = df.sample(n=5000, random_state=42)

# Appliquer le prétraitement sur les colonnes 'Body' et 'Title'
df['Body'] = df['Body'].apply(preprocess_text)
df['Title'] = df['Title'].apply(preprocess_text)

# Combinaison des colonnes 'Title' et 'Body' en une seule colonne 'Text'
df['Text'] = df['Title'] + ' ' + df['Body']

# Prétraiter les tags
df['Tags'] = df['Tags'].str.replace('<', '').str.replace('>', ' ').str.split()

# Sauvegarder le modèle de prétraitement
with open('modelpreprod.pkl', 'wb') as f:
    pickle.dump(preprocess_text, f)


# Fonction pour prédire les tags
def predict_tags(preprocessed_text):
    # Convertir le texte prétraité en une représentation bag-of-words
    bow = dictionary.doc2bow(preprocessed_text.split())
    
    # Utiliser le modèle LDA pour obtenir les topics les plus probables
    topics = lda_model[bow]
    
    # Trier les topics par leur probabilité, et prendre les N topics les plus probables
    sorted_topics = sorted(topics, key=lambda x: x[1], reverse=True)
    top_topics = sorted_topics[:25]  # N est ici 5, mais vous pouvez choisir un autre nombre si vous le souhaitez
    
    # Pour chaque topic, obtenir le mot le plus probable et le traiter comme un "tag"
    tags = []
    for topic in top_topics:
        word, _ = lda_model.show_topic(topic[0], topn=1)[0]  # Obtenir le mot le plus probable pour ce sujet
        tags.append(word)
    
    return tags


# Initialiser le script
init()


