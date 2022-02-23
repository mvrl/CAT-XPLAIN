# This script prepares data for our IMDB sentence experiment

#What it does:
# 1. Downloads IMDB sentiment analysis data from main webpage
# 2. Merges all the data instances with 10 <= #sentences <= 50. 
#    Note: the #sentences > 50 in database is only 0.03% and 
#    to visualize the affect of our experiment we would need at least 10 sentences.
# 3. Perform train/val/test split with ratio 70:10:20 on the overall dataset.


# padding_sentence = "##### ##### ##### ##### #####"

import nltk
from nltk.tokenize import sent_tokenize
import os
from config import imdb_data_path
nltk.download('punkt',download_dir=imdb_data_path)
nltk.data.path.append(imdb_data_path)
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import pandas as pd
from sklearn.model_selection import train_test_split

def split_stratified_into_train_val_test(df_input, stratify_colname='label',
                                         frac_train=0.7, frac_val=0.10, frac_test=0.20,
                                         random_state=42):
    #copied from:
    # https://stackoverflow.com/questions/50781562/stratified-splitting-of-pandas-dataframe-into-training-validation-and-test-set
    

    if frac_train + frac_val + frac_test != 1.0:
        raise ValueError('fractions %f, %f, %f do not add up to 1.0' % \
                         (frac_train, frac_val, frac_test))

    if stratify_colname not in df_input.columns:
        raise ValueError('%s is not a column in the dataframe' % (stratify_colname))

    X = df_input # Contains all columns.
    y = df_input[[stratify_colname]] # Dataframe of just the column on which to stratify.

    # Split original dataframe into train and temp dataframes.
    df_train, df_temp, y_train, y_temp = train_test_split(X,
                                                          y,
                                                          stratify=y,
                                                          test_size=(1.0 - frac_train),
                                                          random_state=random_state)

    # Split the temp dataframe into val and test dataframes.
    relative_frac_test = frac_test / (frac_val + frac_test)
    df_val, df_test, y_val, y_test = train_test_split(df_temp,
                                                      y_temp,
                                                      stratify=y_temp,
                                                      test_size=relative_frac_test,
                                                      random_state=random_state)

    assert len(df_input) == len(df_train) + len(df_val) + len(df_test)

    return df_train, df_val, df_test

def custom_data_prep(storage_path=imdb_data_path, max_count=50, min_count=10):
    splits = ['test','train']
    labels = ['pos','neg']
    reviews_list = []
    labels_list = []
    count_list = []
    for split in splits:
        for label in labels:
            folder_path = os.path.join(storage_path,'aclImdb',split,label)
            files = os.listdir(folder_path)
            for f in tqdm(files):
                filename = os.path.join(folder_path,f)
                with open(filename, 'r') as infile:
                    text = infile.read().strip()
                    sentences = sent_tokenize(text)
                    sent_count = len(sentences)
                    if sent_count >= 10 and sent_count <= 50:
                        reviews_list.append(text)
                        labels_list.append(label)
                        count_list.append(sent_count)
    
    df = pd.DataFrame(columns = ['review','label','sentence_count'])
    df['review'] = reviews_list
    df['label'] = labels_list
    df['sentence_count'] = count_list

    df_train, df_val, df_test = split_stratified_into_train_val_test(df)

    df_train.to_csv(os.path.join(storage_path,'IMDB_train.csv'))
    df_val.to_csv(os.path.join(storage_path,'IMDB_val.csv'))
    df_test.to_csv(os.path.join(storage_path,'IMDB_test.csv'))

    print("Data split done!")


custom_data_prep(storage_path=imdb_data_path, max_count=50, min_count=10)




