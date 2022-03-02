# This script splits the provided trainset into train/val/test splits with ratio 70:10:20

import pandas as pd
from sklearn.model_selection import train_test_split
from config import csv_path
import os

def split_stratified_into_train_val_test(df_input, stratify_colname='Target',
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



train_detail_df = pd.read_csv(os.path.join(csv_path,'stage_2_train_labels.csv'))
df_train, df_valid, df_test = split_stratified_into_train_val_test(train_detail_df)

df_train.to_csv(os.path.join(csv_path,'train_labels.csv'))
df_valid.to_csv(os.path.join(csv_path,'valid_labels.csv'))
df_test.to_csv(os.path.join(csv_path,'test_labels.csv'))

print("Data split done!")


