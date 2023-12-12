# MusicVec - Vector Embeddings for Music (Artists and Songs)
## Table of Contents
I. [Summary](https://github.com/AneeshPatel/MusicVec#i-summary)

II. [What Do Vector Embeddings Even Mean?](https://github.com/AneeshPatel/MusicVec#ii-what-do-vector-embeddings-even-mean)

III. [How Does the Model Work?](https://github.com/AneeshPatel/MusicVec#iii-how-does-the-model-work)

IV. [Using MusicVec](https://github.com/AneeshPatel/MusicVec#iv-using-musicvec)

V. [Future Plans](https://github.com/AneeshPatel/MusicVec#v-future-plans)

VI. [References](https://github.com/AneeshPatel/MusicVec#vi-references)

## I. Summary
This project utilizes vector embeddings to analyze user-generated Spotify playlist data from January 2010 to October 2017 to generate models similar to Word2Vec but tailored for artists (Artist2Vec) and songs (Song2Vec). The infrastructure also allows users to employ their own data for custom model training or to update the pre-trained Artist2Vec and Song2Vec models.

## II. What Do Vector Embeddings Even Mean?
Vector embeddings are a fundamental concept of a unique Natural Language Processing technique (Word2Vec) that examines words based on their context and represents them as numeric vectors. This enables various operations on words, such as finding similar words, identifying outliers in a list of words, calculating similarity percentages between words, and even performing arithmetic operations (addition and subtraction) between words. For instance, in word arithmetic, `KING` - `MAN` + `WOMAN` equals `QUEEN`, and `BIGGER` - `BIG` + `SMALL` equals `SMALLER`. This project extends this idea to artists and songs.

## III. How Does the Model Work?
### A) High-Level
Conceptually, Word2Vec works by learning relationships between words and their context words that appear shortly before or after them in sentences. This model functions similarly, by learning relationships between artists or songs and their neighboring entities within playlists.

### B) Technical
#### i) Dataset and Pre-Processing
The 1 million playlists dataset was preprocessed to iteratively yield one track from a playlist at a time, allowing efficient utilization of the Gensim Python Word2Vec module. For both models, "sentences" were represented by individual playlists, and "words" were represented by either the `artist_name` or the unique `track_uri` for each song, for Artist2Vec and Song2Vec respectively.

`artist_name` was chosen for Artist2Vec because of its immediate interpretability; however, a future step may involve considering the switch to `artist_uri` for a more robust representation of different artists who share the same name.

The choice to use `track_uri` for Song2Vec, rather than just the song name, was motivated by the need to account for potential duplicate song titles across different artists, ensuring an unambiguous representation of individual tracks in the model.

#### ii) Training the Models
Both models share the following hyperparameters:
- `window` of 10
- 7 `workers` threads
- 100 `epochs`
- CBOW training algorithm, chosen for speed and performance with more frequent words

Each model took upwards of 6 hours to complete training on a 2020 M1 MacBook Pro. Due to the extensive training time, different hyperparameters were not experimented with. 

#### iii) Using the Models
Gensim allows trained models to be saved, loaded, and updated with additional data. Both models were saved, and the appropriate one is loaded according to the user's requests. 

##### Artist2Vec
Upon a user query to Artist2Vec, the user input is fed directly to the model and the model output is directly returned, which is possible because the model stores the raw `artist_name` associated with each track during training.

##### Song2Vec
For Song2Vec, the model stores `track_uri`s, which are identifiers that are not immediately interpretable (e.g. spotify:track:6rqhFgbbKwnb9MLmUQDhG6). When a user interacts with Song2Vec, a few additional steps are needed from the model:
1. The user enters the song name, artist name, or both
2. The model queries Spotipy to retrieve the 10 most relevant songs
3. The user selects a song
4. The chosen song's `track_uri` is parsed and passed to the model
5. The model, in turn, queries Spotipy to obtain the track name and artist before returning the final output.

This approach ensures a user-friendly experience when dealing with Song2Vec, accounting for the nature of `track_uri` identifiers and providing meaningful information about the songs users are interested in exploring or analyzing.

## IV. Using MusicVec
1. Clone this repo.
2. Start MusicVec by running `python3 musicvec.py`.
3. Enter your Spotify username and grant access (for the Spotipy API, required by Song2Vec; no personal data is read/pulled)
4. Choose to train a new/existing model or explore an existing model
5. Follow the prompts and enjoy!

> Note: If training a new model, the Spotify 1 Million Playlist Dataset is not included in the repo, but can be found [here](https://www.aicrowd.com/challenges/spotify-million-playlist-dataset-challenge). Also, new models should either define a new class that implements `MusicVecModelInterface` from `models.py` or use one of the existing classes.

## V. Future Plans:
- Obtain Spotify featured playlist data to fill the missing data gap from October 2017 to present
- Leverage the Spotipy API to allow users to update the models using their own playlist data
- Build out a more user-friendly interface
- Enable users to start listening to song or artist with a simple click/button press
- Let users filter by audio features (danceability, loudness, tempo, etc)
- Automatically generate Spotify playlists

## VI. References:
- Spotify Million Playlist Dataset (https://www.aicrowd.com/challenges/spotify-million-playlist-dataset-challenge)
- Spotipy API (https://spotipy.readthedocs.io/en/2.22.1/)
- Tomy Tjandra (https://algoritmaonline.com/song2vec-music-recommender/) for Gensim model training and logging examples
- Radim Rehurek (https://radimrehurek.com/gensim/models/word2vec.html) for Gensim reference docs and tutorials
- Tedboy (https://tedboy.github.io/nlps/generated/generated/gensim.models.Word2Vec.html#gensim.models.Word2Vec) for Gensim Word2Vec function references
