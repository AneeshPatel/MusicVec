#!/usr/bin/env python
"""
This file contains the core code for the MusicVec project.

This project applies vector embeddings to user-generated Spotify playlist data from January 2010 to
October 2017 to generate models similar to Word2Vec but for artists and songs (AKA Artist2Vec and
Song2Vec respectively). It also has the infrastructure to allow users to use their own data to
either train custom models or update the pre-trained Artist2Vec and Song2Vec models.

Future Plans:
- Obtain Spotify featured playlist data to fill the missing data gap from October 2017 to present
- Leverage the Spotipy API to allow users to update the models using their own playlist data
- Build out a more user-friendly interface
- Enable users to start listening to song or artist with a simple click/button press
- Let users filter by audio features (danceability, loudness, tempo, etc)
- Automatically generate Spotify playlists

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
import sys

# Spotipy Related
import spotipy
import spotipy.util as util

# From models.py
from models import Playlists
from models import trainModel, saveModel, loadModel, createEntireModel
from models import MusicVecModelInterface, Artist2VecModel, Song2VecModel


########## USER INTERFACE ##########
print("\n\n========== WELCOME TO MUSICVEC ==========\n\n")
print("~~~~~ About MusicVec ~~~~~")
print(("MusicVec empowers you to dive into the world of music like never before!"
       "\n\nInspired by Word2Vec, this innovative tool utilizes vector embeddings to unlock hidden "
       "connections and unveil the intricacies of music perception. By leveraging data from 1 "
       "million user-created Spotify playlists between January 2010 and October 2017, MusicVec "
       "delves deep into listener preferences, exposing stylistic influences and predicting your "
       "next favorite song."
       "\n\nMusicVec comes with 2 pre-trained models, Artist2Vec and Song2Vec, that allow you to "
       "play around with vector embeddings for individual artists and songs. It also provides a "
       "platform for you to train your own models or update the provided models with your own "
       "data!"))
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
print("\nTo use this model, you will need to log into your Spotify account.")

username = input(">>> Please input your Spotify username: ")
scope = 'playlist-read-private playlist-read-collaborative'

# Erase the cache and prompt for user permission
try:
    token = util.prompt_for_user_token(username, scope)
except:
    os.remove(f".cache-{username}")
    token = util.prompt_for_user_token(username, scope)

# Create spotifyObject
sp = spotipy.Spotify(auth=token)

artist_model = loadModel('models/artist2vec.model')
song_model = loadModel('models/song2vec.model')
Artist2VecModel(artist_model)
Song2VecModel(song_model, sp)

while True:
    print("\n>>> Would you like to train a model or play around with an existing model?")
    task = input("Your choice (0 to train, 1 to play around, x to quit): ").lower()
    if task == '0':
        print("\n>>> Would you like to train a new model from scratch or continue trainingan existing model?")
        version = input("Your choice (0 to train a new model, 1 to continue training an existing model): ")

        if version == '0':
            print("\n>>> Are you sure? Training a new model from scratch can take a very long time.")
            confirm = input("Y/N: ").lower()
            if confirm == 'y':
                print("\n>>> Training a new model requires playlist data in a JSON format.")
                data_folder = input("Where is the data located? Provide the path to the folder containing the files: ")
                feature = input(">>> What feature would you like to train the model on? Examples include 'artist_name' for Artist2Vec and 'track_uri' for Song2Vec: ")
                output_file = input(">>> Where would you like the trained model to be saved? Provide the file path (cannot already exist): ")
                createEntireModel(data_folder, feature, output_file)
        
        if version == '1':
            input_file = input("\n>>> Great! Which model would you like to continue training? Provide the file path: ")
            data_folder = input(">>> Where is the new data located? Provide the path to the folder containing the files: ")
            feature = input(">>> What feature would you like to train the model on? Examples include 'artist_name' for Artist2Vec and 'track_uri' for Song2Vec: ")
            total_examples = int(input(">>> How many total playlists are in the new data? "))
            output_file = input(">>> Where would you like the trained model to be saved? Provide the file path (cannot already exist): ")

            orig_model = loadModel(input_file)
            new_playlists = Playlists(data_folder, feature)
            trained_model = trainModel(orig_model, new_playlists, total_examples=total_examples)
            saveModel(trained_model, output_file)
    
    elif task == '1':
        print("\n>>> Great! You can either play around with one of the provided models (Artist2Vec or Song2Vec) or your own model.")
        model_choice = input("Which model would you like to play around with? Enter 'Artist2Vec', 'Song2Vec', or the file path to your custom model: ")
        model = MusicVecModelInterface.models[model_choice]

        while True:
            print("""\n>>> There are four primary ways in which you can query the model: \
                  \n\n\t  (1) Find the N Most Similar Items: \
                  \n\t\t  Input: an item and a number N (default: 10) \
                  \n\t\t  Output: N most similar items \

                  \n\n\t  (2) Find the Item that Doesn't Match \
                  \n\t\t  Input: list of items \
                  \n\t\t  Output: item that doesn't go with the others \

                  \n\n\t  (3) Similarity Percentage \
                  \n\t\t  Input: two items \
                  \n\t\t  Output: similarity percentage between the items \

                  \n\n\t  (4) Arithmetic \
                  \n\t\t  Input: list of 'positive' items, list of 'negative' items, and a number N (default: 10) \
                  \n\t\t  Output: N most similar items to the sum of the positive items subtracted by the negative items

                  \n ** item refers to the model's feature of interest (ex. artist name for Artist2Vec, song title for Song2Vec) **
                  """)
            choice = input("What would you like to select? (Enter a number 1-4 or x to quit): ").lower()
            
            if choice == "1":
                print(">>> You picked option (1) Find the N Most Similar Items")
                item = model.get_item("What item would you like to find other similar items to? ")
                topn = int(input(">>> How many similar items would you like the model to return? "))
                model.most_similar(item, topn)

            elif choice == "2":
                print(">>> You picked option (2) Find the Item that Doesn't Match")
                item_list = model.getUserList("item")
                model.doesnt_match(item_list)

            elif choice == "3":
                print(">>> You picked option (3) Similarity Percentage")
                item1 = model.get_item("What is the first item? ")
                item2 = model.get_item("What is the second item? ")
                model.similarity(item1, item2)

            elif choice == "4":
                print(">>> You picked option (4) Arithmetic")
                positive_list = model.getUserList("positive item")
                negative_list = model.getUserList("negative item")   
                topn = int(input(">>> How many similar items would you like the model to return? "))      
                model.arithmetic(positive_list, negative_list, topn)

            elif choice == "x":
                break
    elif task == 'x':
        sys.exit(1)
