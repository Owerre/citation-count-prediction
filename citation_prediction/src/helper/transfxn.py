####################################
# Author: S. A. Owerre
# Date modified: 09/06/2021
# Class: Transformations
####################################

# Filter warnings
import warnings
warnings.filterwarnings("ignore")

# Data manipulation
import pandas as pd
import numpy as np

# Text analytics
import nltk
import string
import gensim
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer

# Helps with importing functions from different directory
import os
import sys
BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_PATH)

# Data pre-processing
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Dimensionality reduction
from sklearn.decomposition import PCA

# import custom class
from helper import log_transfxn as cf 

class TransformationPipeline:
    """
    A class for transforming a text dataset
    """
    def __init__(self) -> None:
        pass


    def sen_tokenizer(self, text):
        """
        Sentence tokenizer removes :
        
        1. special characters
        2. punctuations
        3. stopwords
        and finally lemmatizes the token

        Parameters
        ----------
        text:  A string of texts or sentences

        Returns
        ----------
        Lemmatized token 
        """
        my_stop_words = ['use', 'two', 'show', 'study', 'result']
        stop_words = stopwords.words('english')
        stop_words.extend(my_stop_words)
        
        # remove special characters
        symbols = string.punctuation + '0123456789\n'
        nospe_char = [char for char in text if char not in symbols]
        nospe_char = ''.join(nospe_char)
        
        # lower case, tokenize, lemmatizer, and removes top words
        token = nospe_char.lower().split()
        token = [self.word_lemmatizer(x) for x in token if len(x) > 3]
        token = [x for x in token if x not in stop_words]
        return token
    
    def get_wordnet_pos(self, word):
        """
        POS (part of speech) tag to help lemmatizer to be effective.
        For example: goes and going will be lemmatized as go

        Parameters
        ----------
        word:  A word

        Returns
        ----------
        POS tag
        """
        tag = nltk.pos_tag([word])[0][1][0].upper()
        tag_dict = {"J": wordnet.ADJ,
                    "N": wordnet.NOUN,
                    "V": wordnet.VERB,
                    "R": wordnet.ADV}

        return tag_dict.get(tag, wordnet.NOUN)
    
    def word_lemmatizer(self, word):
        """
        Word lemmatization function
        
        Parameters
        ----------
        word:  A word

        Returns
        ----------
        Lemmatized word
        """
        lemmatizer = WordNetLemmatizer()
        word = lemmatizer.lemmatize(word, self.get_wordnet_pos(word))
        return word
    
    def add_bigram(self, token):
        """
        Add bigrams in the data
        """
        bigram = gensim.models.Phrases(token)
        bigram = [bigram[line] for line in token]
        return bigram


    def add_trigram(self, token):
        """
        Add trigrams in the data
        """
        bigram = self.add_bigram(token)
        trigram = gensim.models.Phrases(bigram)
        trigram = [trigram[line] for line in bigram]
        return trigram

    def bow_vector(self, data, text_col):
        """
        Create bag of words vector (i.e. document-term matrix) using CountVectorizer 
        
        Parameters
        ----------
        data:  Pandas dataframe with a text column
        text_col:  Text column in data

        Returns
        ----------
        bow vectors in Pandas Dataframe
        """
        counter = CountVectorizer(tokenizer = self.sen_tokenizer)        
        bow_docs = pd.DataFrame(counter.fit_transform(data[text_col]).toarray(),
                                columns = counter.get_feature_names()
                               )
        vocab = tuple(bow_docs.columns)
        return bow_docs, vocab
    
    def compute_coherence_lda(self, corpus, dictionary, tokens_list, start=None, limit=None, step=None):
        """
        Compute c_v coherence for various number of topics
        """
        topic_coherence = []
        model_list = []
        texts = [[token for sub_token in tokens_list for token in sub_token]]
        for num_topics in range(start, limit, step):
            model = LdaMulticore(corpus=corpus, id2word=dictionary, num_topics=num_topics,
                                 eta='auto', workers=4, passes=20, iterations=100,
                                 random_state=42, eval_every=None,
                                 alpha='asymmetric',  # shown to be better than symmetric in most cases
                                 decay=0.5, offset=64  # best params from Hoffman paper
                                 )
            model_list.append(model)
            coherencemodel = CoherenceModel( model=model, texts=texts, dictionary=dictionary, coherence='c_v')
            topic_coherence.append(coherencemodel.get_coherence())
        return model_list, topic_coherence
    
    def num_pipeline(self, X_train, X_test):
        """
        Transformation pipeline of data with only numerical variables

        Parameters
        ___________
        X_train: Training feature matrix
        X_test: Test feature matrix

        Returns
        __________
        Transformation pipeline and transformed data in array
        """
        # Create pipeline
        num_pipeline = Pipeline([ ('log', cf.LogTransformer()),
                                ('std_scaler', StandardScaler()),
                                ])

        # Original numerical feature names 
        feat_nm = list(X_train.select_dtypes('number'))

        # Fit transform the training set
        X_train_scaled = num_pipeline.fit_transform(X_train)
        
        # Only transform the test set
        X_test_scaled = num_pipeline.transform(X_test)
        return X_train_scaled, X_test_scaled, feat_nm
    
    def cat_encoder(self, X_train, X_test):
        """
        Transformation pipeline of categorical variables

        Parameters
        ___________
        X_train: Training feature matrix
        X_test: Test feature matrix

        Returns
        __________
        Transformation pipeline and transformed data in array
        """
        # Instatiate class
        one_hot_encoder = OneHotEncoder()

        # Fit transform the training set
        X_train_scaled = one_hot_encoder.fit_transform(X_train)
        
        # Only transform the test set
        X_test_scaled = one_hot_encoder.transform(X_test)

        # Feature names for output features
        feat_nm = list(one_hot_encoder.get_feature_names(list(X_train.select_dtypes('O'))))
        return X_train_scaled.toarray(), X_test_scaled.toarray(), feat_nm
  
    def preprocessing(self, X_train, X_test):
        """
        Transformation pipeline of data with both numerical and categorical 
        variables.

        Parameters
        ___________
        X_train: Training feature matrix
        X_test: Test feature matrix

        Returns
        __________
        Transformed data in array
        """

        # Numerical transformation pipepline
        num_train, num_test, num_col = self.num_pipeline(X_train.select_dtypes('number'), 
                                        X_test.select_dtypes('number'))

        # Categorical transformation pipepline
        cat_train, cat_test, cat_col = self.cat_encoder(X_train.select_dtypes('O'), 
                                        X_test.select_dtypes('O'))

        # Transformed training set
        X_train_scaled = np.concatenate((num_train,cat_train), axis = 1)

        # Transformed test set
        X_test_scaled = np.concatenate((num_test,cat_test), axis = 1)

        # Feature names
        feat_nm = num_col + cat_col
        return X_train_scaled, X_test_scaled, feat_nm
    
    def pca_plot_labeled(self, data_, labels, palette = None):
        """
        Dimensionality reduction of labeled data using PCA 

        Parameters
        __________
        data: scaled data
        labels: labels of the data
        palette: color list

        Returns
        __________
        Matplotlib plot of two component PCA
        """
        #PCA
        pca = PCA(n_components = 2)
        X_pca = pca.fit_transform(data_)

        # put in dataframe
        X_reduced_pca = pd.DataFrame(data = X_pca)
        X_reduced_pca.columns = ['PC1', 'PC2']
        X_reduced_pca['class'] = labels.reset_index(drop = True)

        # plot results
        plt.rcParams.update({'font.size': 15})
        plt.subplots(figsize = (12,8))
        sns.scatterplot(x = 'PC1', y = 'PC2', data = X_reduced_pca,
        hue = 'class', palette = palette)

        # axis labels
        plt.xlabel("Principal component 1")
        plt.ylabel("Principal component 2")
        plt.title("Dimensionality reduction")
        plt.legend(loc = 'best')
        plt.savefig('../images/pca.png')
        plt.show()
    
