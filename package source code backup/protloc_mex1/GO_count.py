# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 18:24:37 2023

@author: qq102
"""



import pandas as pd

import re

from functools import reduce
from packaging import version
import warnings
try:
    import gensim
    if version.parse(gensim.__version__) < version.parse('4.2.0'):
        warnings.warn("Your gensim version is older than 4.2.0  +\
                      and may be not operation correct.")
except ImportError:
    raise ValueError('gensim is not installed so this module is not available! Run `pip install gensim==4.2.0` " \
                             "to fix this ')

## Building sentence processing functions
def read_corpus(sequence_input, tokens_only=False):
    '''
    
    
    Reads input text data and yields tokenized sentences or tagged sentences.
    
    Parameters
    ----------
    sequence_input : list of strings
        The input text data with per sentence.
    tokens_only : bool, optional
        If True, yields tokenized sentences only. If False, yields tagged sentences
        with sentence index as the tag. The default is False.
    Yields
    ------
    list of str or gensim.models.doc2vec.TaggedDocument
        If tokens_only is True, yields a simple preprocess for each sentence with
        no the sentence index as the tag,
        
        If tokens_only is False, yields a TaggedDocument object for each sentence
        with the sentence index as the tag. The TaggedDocument object contains
        the tokenized sentence.

    '''
    for i, line in enumerate(sequence_input):
        tokens = gensim.utils.simple_preprocess(line)
        if tokens_only:
            yield tokens
        else:
            # For training data, add tags
            yield gensim.models.doc2vec.TaggedDocument(tokens, [i])



## Building GO language string processing functions
def GO_pre_Process(GO_data,pattern_GO='\\[GO:\d+\\]',GO_name='GO_BP'):
    '''
    
    
    Pre-processes GO annotation data by removing GO ontology terms.
    
    Parameters
    ----------
    GO_data : DataFrame
        The input GO annotation data.
    pattern_GO : str, optional
        The regular expression pattern for matching GO ontology terms.
        The default is '\\[GO:\d+\\]' to match terms like [GO:0006397].
    GO_name : str, optional
        The name of the column containing GO ontology terms.
        The default is 'GO_BP'.
    Returns
    -------
    GO_data : DataFrame
        The GO annotation data with GO ontology terms (such as [GO:0006397]) 
        removed from the GO_name column.

    '''
    pattern=re.compile(pattern_GO)
    ## Removing [GO:XXXXXX] from bp, Step 1 
    for i in range(len(GO_data[GO_name])):
        GO_data.loc[i,GO_name]=re.sub(pattern,"",GO_data.loc[i,GO_name])
    return GO_data

## Building doc Embedding Generation Functions

def model_training(file_Corpus_input,model,sequence_name='GO_BP'):
    '''
    

    Trains a gensim document embedding model on input corpus data.
    
    Parameters
    ----------
    file_Corpus_input : DataFrame
        The input corpus data.
    model : gensim.models.doc2vec model
        The document embedding model to be trained.
    sequence_name : str, optional
        The name of the column containing training sentences in file_Corpus_input.
        The default is 'GO_BP'.
    Returns
    -------
    model : gensim.models.doc2vec model
        The trained document embedding model
    Note
    -------
    total_examples=model.corpus_count 
    This means we pass the value of model.corpus_count as the total_examples parameter to the model.train() function.
    The total_examples parameter specifies the total number of documents the model will train on. 
    This allows the algorithm to allocate the necessary memory in advance and improve training efficiency. 
    model.corpus_count should be the number of documents or sentences in our training corpus corpus_data. 
    So here we directly use the corpus size statistics of the model itself as the value of total_examples. 
    If total_examples is not specified, Gensim will automatically infer the corpus size, 
    but this may slightly reduce training efficiency because it has to scan the corpus twice - once to get the size statistics and once for actual training.
    In addition, the function does not support single-character sentences at now
    '''
    corpus_data=list(read_corpus(file_Corpus_input[sequence_name]))
    # Creating a vocabulary using only the corpus
    
    model.build_vocab(corpus_data)
    
    # Computing Word Embeddings from the Corpus
    model.train(corpus_data, total_examples=model.corpus_count, epochs=model.epochs)
    return model

def vec_create_GO(X_input,model,sequence_name='GO_BP',model_inference=0.5):
    '''
    

    Generates document embedding vectors for input data using the pre-trained document embedding model.
    
    Parameters
    ----------
    X_input : DataFrame
        The input data for which to generate document embeddings.
    model : gensim.models.doc2vec model
        The pre-trained document embedding model to use for generating inference embeddings.
    sequence_name : str, optional
        The name of the column in X_input containing training sentences. The default is 'GO_BP'.
    model_inference : float, optional
        The proportion of additional epochs to train the model on the new data.
        The default is 0.5.
    Raises
    ------
    ValueError
        If X_input contains duplicate index.
    Returns
    -------
    train_vector_data : DataFrame
        The document embedding vectors for X_input indexed by the same index.

    '''
    if X_input.index.duplicated().sum() > 0:
        raise ValueError("Index contains duplicates")
    ## Processing sentences in X_train and X_test
    train_corpus = list(read_corpus(X_input[sequence_name],tokens_only=True))
  
    
    # Converting data into feature embeddings

    train_vector=[]
    for doc_id in range(len(train_corpus)):
        inferred_vector=model.infer_vector(train_corpus[doc_id],
                                       epochs=int(model.epochs+model_inference*model.epochs))
        train_vector.append(inferred_vector)

  
    ## Transposing data into a pandas DataFrame

    for i,value in enumerate(train_vector):
        train_vector[i]=pd.DataFrame(value).T
    
    train_vector_data=reduce(lambda x,y:pd.concat([x,y],axis=0),train_vector)
    
 

    train_vector_data.index=X_input.index
    train_vector_data.columns=[sequence_name+str(i) for i in train_vector_data.columns]
    
    return train_vector_data



def AA_combine(X_data, sequence_name, number=3):
    '''
    

    Combines every n amino acids in a sequence into a single character.
    
    Parameters
    ----------
    X_data : DataFrame
        The input data containing an amino acid sequence column.
    sequence_name : str
        The name of the column in X_data containing the amino acid sequence.
    number : int, optional
        The number of amino acids to combine into a single character.
        The default is 3.
    Returns
    -------
    X_data : DataFrame
        The input data with the amino acid sequence column modified to
        contain combined amino acid characters.


    '''
    X_data = X_data.copy()

    def combine_sequence(value):
        # Initializing an empty sequence string 
        sequence_inner = ""
        # Iterating over an amino acid sequence
        
        for i in range(0, len(value), number):
            # Extracting 3 amino acids
            triplet = value[i:i+number]
            # Concatenating 3 amino acids into one character and adding spaces
            sequence_inner += triplet + " "

        return sequence_inner

    X_data[sequence_name] = X_data[sequence_name].apply(combine_sequence)

    return X_data