import re
import os
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
import gensim
from gensim.models import LdaModel
from gensim.corpora import Dictionary


nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('stopwords')

# Initialisation globale du lemmatiseur, du modèle LDA et du dictionnaire
lemmatizer = None
lda_model = None
dictionary = None

def load_resources():
    global lda_model, dictionary

    # Chargement des ressources à partir de fichiers
    with open(os.path.join(os.path.dirname(__file__), 'dictionary.pkl'), 'rb') as f:
        dictionary = pickle.load(f)

    with open(os.path.join(os.path.dirname(__file__), 'lda_model.pkl'), 'rb') as f:
        lda_model = pickle.load(f)

# Appeler la fonction pour charger les ressources
load_resources()


def init():
    global lemmatizer, lda_model, dictionary

    # Télécharger les ressources nécessaires
    nltk.download('averaged_perceptron_tagger')
    nltk.download('wordnet')
    nltk.download('stopwords')

    # Initialiser le lemmatiseur de NLTK
    lemmatizer = WordNetLemmatizer()





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
df = pd.read_csv("C:/Users/matth/Documents/stack/QueryResults.csv")

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


# Préparation des données
texts = df['Text'].apply(lambda x: x.split()).tolist()
dictionary = gensim.corpora.Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]

# Entraînement du modèle LDA avec des hyperparamètres supplémentaires
lda_model = gensim.models.LdaModel(
    corpus=corpus,
    id2word=dictionary,
    num_topics=50,
    passes=5,
    iterations=5,  # Ajout de l'hyperparamètre 'iterations'
   
)

# Sauvegarde du modèle LDA
with open('lda_model.pkl', 'wb') as f:
    pickle.dump(lda_model, f)

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


