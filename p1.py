import pandas as pd
import numpy as np
import scipy as sp
from gensim.test.utils import datapath
from gensim.models import KeyedVectors


# from gensim.test.utils import lee_corpus_list
# from gensim.models import Word2Vec

def ReadDataSetSimLex999():
    return pd.read_csv('SimLex-999.txt', sep="\t")


def euclideanDisctance(vec_word_1, wec_word_2):   #Input: two vectors, Output: euclidian distance (scalar)
    return np.sqrt(np.sum((vec_word_1 - wec_word_2) ** 2))

#Initiates zero vectors before overwriting them
def checkWordInWord2Vec(wv_from_bin, word1, word2):
    word1_vec = np.zeros(300)
    word2_vec = np.zeros(300)

    #Check wether the word1 and word2 are known by the model, so it can return vector per word: wv_from_bin.get_vector(word1)
    if wv_from_bin.__contains__(word1):
        word1_vec = wv_from_bin.get_vector(word1)

    if wv_from_bin.__contains__(word2):
        word2_vec = wv_from_bin.get_vector(word2)

    return euclideanDisctance(word1_vec,word2_vec)

def main():
    simlex999 = ReadDataSetSimLex999()

    # 1.2 Data Reader read the calculate the distances between the pairs (hard, easy), (hard, difficult) and (hard, dense)
    print('1.2')
    print('hard easy', simlex999.loc[
              (simlex999['word1'] == 'hard')
              &
              (simlex999['word2'] == 'easy')
              ]['SimLex999'].to_numpy().flatten()[0]
          )

    print('hard difficult', simlex999.loc[
              (simlex999['word1'] == 'hard')
              &
              (simlex999['word2'] == 'difficult')
              ]['SimLex999'].to_numpy().flatten()[0]
          )

    print('hard dense', simlex999.loc[
              (simlex999['word1'] == 'hard')
              &
              (simlex999['word2'] == 'dense')
              ]['SimLex999'].to_numpy().flatten()[0]
          )

    # 1.3 Ranking Based on word2vec

    # load bin
    file_name = 'GoogleNews-vectors-negative300.bin'
    wv_from_bin = KeyedVectors.load_word2vec_format(file_name, binary=True)

    euclidean_distance_of_words = pd.DataFrame([
        checkWordInWord2Vec(wv_from_bin, # model
                            row['word1'], # word1 i.e. hard
                            row['word2']) # word2 i.e. easy
        for idx, row in simlex999.iterrows() # loop for each row in our SimLex999 DataSet
    ])

    print('\n1.3')
    print("hard easy", euclidean_distance_of_words.loc[
              (simlex999['word1'] == 'hard')
              &
              (simlex999['word2'] == 'easy')
              ].to_numpy().flatten()[0]
          )

    print("hard difficult", euclidean_distance_of_words.loc[
              (simlex999['word1'] == 'hard')
              &
              (simlex999['word2'] == 'difficult')
              ].to_numpy().flatten()[0]
          )

    print("hard dense", euclidean_distance_of_words.loc[
              (simlex999['word1'] == 'hard')
              &
              (simlex999['word2'] == 'dense')
              ].to_numpy().flatten()[0]
          )

    # 1.4 Correlation
    print('\n1.4')
    print('Pearson’s correlation coefficient: ',
        sp.stats.pearsonr(simlex999['SimLex999'].to_numpy().flatten(),
                          euclidean_distance_of_words.to_numpy().flatten()
                          )[0])
    print('')

if __name__ == "__main__":
    main()
