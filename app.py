from flask import Flask, render_template, request
import preprocessing_code 
import nltk
import pickle
# General Libraries
import gensim
import os
import re
import string
from datetime import timedelta
import datetime as dt

# Data Manipulation Libraries
import numpy as np
import pandas as pd

# Visualization Libraries
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Circle
from bs4 import BeautifulSoup

# Machine Learning Libraries
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, MultiLabelBinarizer, QuantileTransformer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer, KNNImputer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, silhouette_score, accuracy_score, make_scorer
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso, ElasticNet, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.dummy import DummyRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from scipy.stats import f_oneway, f
from scipy import stats
from scipy.sparse import csr_matrix
from scipy.cluster.hierarchy import dendrogram, linkage, ward, fcluster
import scipy.cluster.hierarchy as shc
import scipy.stats as st



# Natural Language Processing Libraries
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet, stopwords, names
from nltk import pos_tag, word_tokenize


# Topic Modelling Libraries

from gensim.models import CoherenceModel
from gensim.corpora import Dictionary
from gensim.models import LdaModel
import gensim
from sklearn.decomposition import LatentDirichletAllocation
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis
from sklearn.decomposition import LatentDirichletAllocation
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis

# Data Persistency
import pickle
import spacy.cli
spacy.cli.download("en_core_web_sm")

nlp = spacy.load("en_core_web_sm")

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')


# Appel de la fonction init() pour charger le modèle LDA et le dictionnaire
preprocessing_code.init()

# Création d'une instance de l'application Flask
app = Flask(__name__)

# Initialiser le module de prétraitement au démarrage de l'application
preprocessing_code.init()

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        question = request.form['question']
        tags = preprocessing_code.predict_tags(question)
        return render_template('results.html', question=question, tags=tags)
    return render_template('form.html')

if __name__ == '__main__':
    app.run(debug=True)