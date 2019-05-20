from flask import Flask, render_template, request
import numpy as np
import pandas as pd
#import os
#import nltk
from nltk.tokenize import sent_tokenize      #common
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx

# Only perform the below operation once
'''
# Downloading nltk data
nltk.download('punkt')
nltk.download('stopwords')
# Word embedding
os.system('wget http://nlp.stanford.edu/data/glove.6B.zip')
os.system('unzip glove*.zip')
os.system('touch glove.6B.txt')
# Appending all files into a single file
os.system('glove.6B.50d.txt glove.6B.100d.txt glove.6B.200d.txt glove.6B.300d.txt > glove.6B.txt')
word_embeddings = {}
f = open('glove.6B.txt', encoding='utf-8')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    word_embeddings[word] = coefs
f.close()
np.save('embeddings.npy', word_embeddings)
'''
# Loading embeddings
word_embeddings = np.load('embeddings.npy')
word_embeddings = word_embeddings[()]

# Creating Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/Trynow')
def summarize():
    return render_template('summarize.html')

@app.route('/result',methods = ['POST', 'GET'])
def result():
    if request.method == 'POST':
        try:
            data = request.form['text']
            n = request.form['number']
            # Preprocessing the text.
            sentences = sent_tokenize(data)
            # make alphabets lowercase
            clean_sentences = [s.lower() for s in sentences]
            # remove punctuations, numbers and special characters
            clean_sentences = pd.Series(clean_sentences).str.replace("[^a-zA-Z]", " ")
            sentence_vectors = []
            for i in clean_sentences:
                if len(i) != 0:
                    v = sum([word_embeddings.get(w, np.zeros((300,))) for w in i.split()])/(len(i.split())+0.001)
                else:
                    v = np.zeros((300,))
                sentence_vectors.append(v)   
            # similarity matrix
            sim_mat = np.zeros([len(sentences), len(sentences)])
            for i in range(len(sentences)):
                for j in range(len(sentences)):
                    if i != j:
                        sim_mat[i][j] = cosine_similarity(sentence_vectors[i].reshape(1,300), sentence_vectors[j].reshape(1,300))[0,0]

            nx_graph = nx.from_numpy_array(sim_mat)
            scores = nx.pagerank(nx_graph)
            ranked_sentences = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True)
            # Generating Summary
            result = []        
            for answer in ranked_sentences:
                result.append(answer[1])    
            result = " ".join(result[:int(n)])  
            print(n)
        except:
            result = "Could not understand the text."
        finally:
            return render_template('result.html', result=result)
if __name__ == '__main__':
    app.run(debug=True)