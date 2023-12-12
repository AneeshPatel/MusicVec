#!/usr/bin/env python
"""
This file contains the Class definitions and helper functions of the MusicVec model.
It is unlikely that a user will ever have to touch this file, and most interactions should be with
the musicvec.py file directly.

Author: Aneesh Patel
Date Last Modified: 12/09/2023
Context: UC Berkeley Music 108 Fall 2023 Final Project

References:
- Spotify Million Playlist Dataset (https://www.aicrowd.com/challenges/spotify-million-playlist-dataset-challenge) 
- Spotipy API (https://spotipy.readthedocs.io/en/2.22.1/)
- Tomy Tjandra (https://algoritmaonline.com/song2vec-music-recommender/) for Gensim model training
    and logging examples
- Radim Rehurek (https://radimrehurek.com/gensim/models/word2vec.html) for Gensim reference docs
    and tutorials
- Tedboy (https://tedboy.github.io/nlps/generated/generated/gensim.models.Word2Vec.html#gensim.models.Word2Vec)
    for Gensim Word2Vec function references
- StackOverflow for misc guidance
"""

# Standard Library
import os

# Gensim Model Related
import multiprocessing
import logging
import time
import dill
from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec
from contextlib import contextmanager

# Misc
import json
from abc import ABC, abstractmethod



########## PLAYLISTS CLASS ##########
class Playlists(object):
    """
    This class stores Playlist data and streams it lazily, one playlist at a time, which
    allows the model to be trained on very large datasets.
    """
    def __init__(self, dirname, feature):
        self.dirname = dirname
        
        # Feature is the feature that we training on ('artist_name', 'track_name')
        self.feature = feature
    
    def __iter__(self):
        for fname in os.listdir(self.dirname):
            data = json.load(open(os.path.join(self.dirname, fname)))
            for p in data['playlists']:
                yield [t[self.feature] for t in p['tracks']]



########## MODEL CLASS ##########
class MusicVecModelInterface(ABC):
    """ Abstract base class interface representing MusicVec models. """

    models = dict() # Mapping of all MusicVec model names to models

    @abstractmethod
    def __init__(self, name, model):
        self.name = name
        self.model = model
        self.models[name] = self

    @abstractmethod
    def get_item(self, prompt):
        """ Prompts the user for an item. Returns the item if provided, else False. """
        pass

    @abstractmethod
    def most_similar(self, item, topn):
        """ Given an item, return the topn-most similar items. """
        pass
    
    @abstractmethod
    def doesnt_match(self, item_list):
        """ Given a list of items, return the item that doesn't match with the others. """
        pass

    @abstractmethod
    def similarity(self, item1, item2):
        """ Given two items, return the similarity percentage between them. """
        pass
    
    @abstractmethod
    def arithmetic(self, positive_items, negative_items, topn):
        """
        Given a list of positive items, list of negative items, return the topn-most
        similar items to the sum of the positive items subtracted by the negative items.
        """
        pass

    def getUserList(self, item_name):
        """"
        Gets a list of inputs from the user of potentially indefinite length. Once get_item() 
        returns False, the list ends and is returned.
        """
        item_list = []
        item = self.get_item(f"Enter {item_name} #{len(item_list) + 1} or nothing if you are done: ")
        while item:
            item_list.append(item)
            item = self.get_item(f"Enter {item_name} #{len(item_list) + 1} or nothing if you are done: ")
        return item_list
    
    def error_msg(self, e):
        return e
        
class Artist2VecModel(MusicVecModelInterface):
    """ Class representing Artist2Vec model. """
    def __init__(self, model):
        super().__init__("Artist2Vec", model)

    def get_item(self, prompt):
        item = input(">>> " + prompt)
        if not item:
            return False
        return item
    
    def most_similar(self, item, topn):
        with handle_exceptions(self):
            output = self.model.wv.most_similar(positive=[item], topn=topn)
            printMostSimilarOutput(output)
    
    def doesnt_match(self, item_list):
        with handle_exceptions(self, multiple_items=True):
            print(self.model.wv.doesnt_match(item_list) + " doesn't match the rest!")
    
    def similarity(self, item1, item2):
        with handle_exceptions(self, multiple_items=True):
            output = self.model.wv.similarity(item1, item2)
            print(f"{item1} and {item2} are {round(output * 100, 2)}% similar!")
    
    def arithmetic(self, positive_items, negative_items, topn):
        with handle_exceptions(self, multiple_items=True):
            output =  self.model.wv.most_similar(positive=positive_items, negative=negative_items, topn=topn)
            printMostSimilarOutput(output)

class Song2VecModel(MusicVecModelInterface):
    """ Class representing Song2Vec model. """
    def __init__(self, model, sp):
        super().__init__("Song2Vec", model)
        self.sp = sp

    def get_item(self, prompt):
        while True:
            query = ""
            print(">>> " + prompt)
            track = input("   >>> Track choice (or empty if none): ")
            artist = input("   >>> Artist choice (or empty if none): ")
            if track:
                query += "track:" + track + " "
            if artist:
                query += "artist:" + artist
            if not query:
                return False
            output = self.sp.search(q=query, type="track")
            printSpotipyQueryOutput(output)
            choice = input(">>> Which track would you like to select? ")
            if choice:
                return output['tracks']['items'][int(choice)]['uri']
    
    def error_msg(self, e):
        first_index = e.find("'")
        second_index = e.find("'", first_index + 1)
        track_uri = e[first_index + 1:second_index]
        return getTrackNameAndArtists(self.sp.track(track_uri)) + e[second_index + 1:]
        
    def most_similar(self, item, topn):
        with handle_exceptions(self):
            output = self.model.wv.most_similar(positive=[item], topn=topn)
            func = lambda track_uri: getTrackNameAndArtists(self.sp.track(track_uri))
            printMostSimilarOutput(output, func=func)

    def doesnt_match(self, item_list):
        with handle_exceptions(self, multiple_items=True):
            track_uri = self.model.wv.doesnt_match(item_list)
            print(getTrackNameAndArtists(self.sp.track(track_uri)) + " doesn't match the rest!")
    
    def similarity(self, item1, item2):
        with handle_exceptions(self, multiple_items=True):
            output = self.model.wv.similarity(item1, item2)
            print(f"{getTrackNameAndArtists(self.sp.track(item1))} and {getTrackNameAndArtists(self.sp.track(item2))} are {round(output * 100, 2)}% similar!")

    def arithmetic(self, positive_items, negative_items, topn):
        with handle_exceptions(self, multiple_items=True):
            output =  self.model.wv.most_similar(positive=positive_items, negative=negative_items, topn=topn)
            func = lambda track_uri: getTrackNameAndArtists(self.sp.track(track_uri))
            printMostSimilarOutput(output, func=func)



########## LOGGING LOGISTICS ##########
logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO)

class Callback(CallbackAny2Vec):
    def __init__(self):
        self.epoch = 1
        self.training_loss = []

    def on_epoch_end(self, model):
        loss = model.get_latest_training_loss()
        if self.epoch == 1:
            current_loss = loss
        else:
            current_loss = loss - self.loss_previous_step
        print(f"Loss after epoch {self.epoch}: {current_loss}")
        self.training_loss.append(current_loss)
        self.epoch += 1
        self.loss_previous_step = loss



########## MODEL FUNCTIONS ##########
def makeModel(playlists):
    """
    Makes and returns a model on 'playlists', a Playlists object representing the
    inputted playlist data.
    """
    print('\n\n\n>>> MAKING MODEL: ', time.ctime(time.time()), '\n\n\n')
    model = Word2Vec(playlists,
                    window=10,
                    sg=0,
                    workers=multiprocessing.cpu_count() - 1)
    print('\n\n\n>>> FINISHED MAKING MODEL: ', time.ctime(time.time()))
    return model

def buildVocab(model, playlists):
    """
    Builds the vocab for 'model' using the 'playlists' data. Returns the model.
    """
    print('\n\n\n>>> BUILDING VOCAB: ', time.ctime(time.time()), '\n\n\n')
    logging.disable(logging.NOTSET) # Enable logging
    model.build_vocab(playlists)
    print('\n\n\n>>> FINISHED BUILDING VOCAB: ', time.ctime(time.time()))
    return model

def trainModel(model, playlists, **kwargs):
    """"
    Trains 'model' using the 'playlists' data. Lazily streams data from 'playlists'
    one playlist at a time. Returns the trained model.
    """
    print('\n\n\n>>> TRAINING MODEL: ', time.ctime(time.time()), '\n\n\n')
    logging.disable(logging.INFO) # Disable logging
    callback = Callback() # Instead, print out loss for each epoch
    model.train(playlists,
                total_examples = kwargs.get('total_examples', model.corpus_count),
                epochs = 100,
                compute_loss = True,
                callbacks = [callback])
    print('\n\n\n>>> DONE TRAINING MODEL: ', time.ctime(time.time()))
    return model

def saveModel(model, output_file):
    """"
    Saves 'model' to 'output_file'. Assumes 'output_file' does not already exist.
    """
    print('\n\n\n>>> SAVING MODEL: ', time.ctime(time.time()), '\n\n\n')
    open(output_file, 'x')
    with open(output_file, 'wb') as f:
        dill.dump(model, f)
    print('\n\n\n>>> DONE SAVING MODEL: ', time.ctime(time.time()))

def loadModel(input_file, verbose=False):
    """ Loads and returns model from 'input_file'. """
    if verbose:
        print('\n\n\n>>> LOADING MODEL: ', time.ctime(time.time()), '\n\n\n')
        logging.disable(logging.NOTSET) # Enable logging
    else:
        logging.disable(logging.INFO) # Disable logging
    model = Word2Vec.load(input_file)
    if verbose:
        print('\n\n\n>>> DONE LOADING MODEL: ', time.ctime(time.time()), '\n\n\n')
    return model

def createEntireModel(data_folder, feature, output_file):
    """
    Creates a full model on the data in 'data_folder', using the 'feature'
    as the parameter of interest, and saves the final model to 'output_file'.

    Example function call for Artist2Vec:
        trainModel(
            'spotify_million_playlist_dataset/data',
            'artist_name',
            'models/artist2vec.model')

    Example function call for Song2Vec: 
        trainModel(
            'spotify_million_playlist_dataset/data',
            'track_uri',
            'models/song2vec.model')
    """
    playlists = Playlists(data_folder, feature)
    initial_model = makeModel(playlists)
    vocab_model = buildVocab(initial_model, playlists)
    trained_model = trainModel(vocab_model, playlists)
    saveModel(trained_model, output_file)



########## MISC HELPER FUNCTIONS ##########
def printMostSimilarOutput(output, func=lambda x:x):
    """
    Prints the 'output' from a call to most_similar() in a more readable format.
    """
    for i in range(len(output)):
        item = func(output[i][0])
        percentage = output[i][1]
        print(f"\t({i + 1}) {item} - {round(percentage * 100, 2)}% similar")

@contextmanager
def handle_exceptions(model, multiple_items=False):
    try:
        yield
    except KeyError as e:
        print("Uh oh! That item does not exist in the dataset.")
        if (multiple_items):
            print("Here is the error: " + model.error_msg(str(e)))
        print("Try again and keep in mind the provided models only have data up till 2017.")
    except:
        print("Unknown error.")

def getTrackNameAndArtists(track, tuple=False):
    track_name = track['name']
    track_year = int(track['album']['release_date'][:4])
    artists = [a['name'] for a in track['artists']]
    if tuple:
        return track_name, artists, track_year
    return track_name + " by " + ", ".join(artists)

def printSpotipyQueryOutput(output):
    for i in range(len(output['tracks']['items'])):
        track = output['tracks']['items'][i]
        track_name, artists, track_year = getTrackNameAndArtists(track, tuple=True)
        print("\n\t(" + str(i) + ") Track Name:", track_name)
        print("\t    Artist(s):", ', '.join(artists))
        if (track_year >= 2017):
            print("\t    (This was released after 2017, so it may not be included in the dataset.)")
