#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import re
import os
from datetime import datetime

import pyspark.sql.functions as F
from pyspark.ml.recommendation import ALS
from pyspark.sql import SparkSession
from pyspark.sql.window import Window
from pyspark.mllib.evaluation import RankingMetrics

def calculateSongListens(df):
    # This function calculates the number of times a user has listened to a song
    # The song_listens column acts as our 'rating' for the recommendation algorithm
    df = df.groupBy('user_id', 'recording_msid').agg(F.count('*').alias('song_listens'))
    return df

def create_lookup(train, valid, test):
    # We create a universe of all the songs and generate a running row_number to assign a unique index to each song
    # This will help us consistently map all songs to a Numeric value among all three datasets, which can be used in the ALS model
    train_id = train.select('recording_msid')
    valid_id = valid.select('recording_msid')
    test_id = test.select('recording_msid')
    song_ids = train_id.union(valid_id).union(test_id).distinct()
    song_ids = song_ids.withColumn('temp', F.lit(1))  # To group over the whole dataset
    song_ids = song_ids.withColumn('recording_msid_idx', F.row_number().over(Window.partitionBy('temp').orderBy('recording_msid')))
    song_ids = song_ids.withColumnRenamed('recording_msid', 'msid').drop('temp')
    return song_ids


def main(spark, file_path):
    '''Main routine for run for Storage optimization template.
    Parameters
    ----------
    spark : SparkSession object
    '''
    
    predictions = 100

    # Hyperparameter tuning on small dataset
    alphas = [0.01, 0.1, 1]
    regs = [0.01, 0.1, 1]
    ranks = [10, 20, 30, 40]

    # Overwrite and choose only one value for running the full dataset
    alphas = [1]
    regs = [0.1]
    ranks = [30]

    time_start = datetime.now()

    train_data = spark.read.parquet(file_path + '/training_all.parquet')
    valid_data = spark.read.parquet(file_path + '/validation_all.parquet')

    # Following filter is additionally added to modify the full dataset 
    # We are filtering the data fulter to keep popular songs with the intuition that popular songs are atleast listened to be 100 distinct users
    # Songs that don't have at least 100 distinct users will not be recommended to users
    good_songs = train_data.groupBy('recording_msid').agg(F.countDistinct("user_id").alias("unique_users"))\
                           .withColumnRenamed('recording_msid', 'temp')
    good_songs = good_songs.filter(good_songs['unique_users'] > 100)
    train_data = train_data.join(good_songs, train_data['recording_msid'] == good_songs['temp'], 'inner').drop('temp')
    print("Filtered records")

    # Read test data to check scores on unseen data
    test_data = spark.read.parquet('/user/bm106_nyu_edu/1004-project-2023/interactions_test.parquet')
    
    train_data = calculateSongListens(train_data)
    valid_data = calculateSongListens(valid_data)
    test_data = calculateSongListens(test_data)
    print("Calculated song listens")

    # Add the numeric song indexing column to all three datasets, which will be used to generate recommendations
    song_lookup = create_lookup(train_data, valid_data, test_data)
    train_data = train_data.join(song_lookup, (train_data['recording_msid'] == song_lookup['msid']), 'left').drop('msid')
    valid_data = valid_data.join(song_lookup, (valid_data['recording_msid'] == song_lookup['msid']), 'left').drop('msid')
    test_data = test_data.join(song_lookup, (test_data['recording_msid'] == song_lookup['msid']), 'left').drop('msid')
    print("Created song index columns")

    for alpha in alphas:
        for reg in regs:
            for rank in ranks:
                # Create ALS model with hyperparameters
                als = ALS(rank=rank, regParam=reg, alpha=alpha, maxIter=5, userCol="user_id", itemCol="recording_msid_idx", ratingCol="song_listens")
                model = als.fit(train_data)
                print("Model fitted")

                # Generate 100 recommendations for all users
                userRecs = model.recommendForAllUsers(predictions)
                print("Recommendations generated")

                userRecs = userRecs.withColumn('pred', F.col('recommendations.recording_msid_idx'))

                test = test_data.groupBy('user_id').agg(F.collect_set('recording_msid_idx').alias('true'))
                recs = userRecs.select('user_id', 'pred')

                test_rdd = test.rdd.map(lambda row: (row['user_id'], row['true']))
                recs_rdd = recs.rdd.map(lambda row: (row['user_id'], row['pred']))
                joined_rdd = test_rdd.join(recs_rdd)
                print("RDDs Joined")

                eval_rdd = joined_rdd.map(lambda row: (row[1][1], row[1][0]))
                metrics = RankingMetrics(eval_rdd)
                
                # Calculate evaluation metrics
                map_val = metrics.meanAveragePrecisionAt(predictions)
                ap_val = metrics.precisionAt(predictions)
                print("Alpha: %f, Reg: %f, Rank: %f" % (alpha, reg, rank))
                print("MAP: %f, AP: %f" % (map_val, ap_val))

    print(datetime.now() - time_start)
        
# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('part2').getOrCreate()

    #If you wish to command line arguments, look into the sys library(primarily sys.argv)
    #Details are here: https://docs.python.org/3/library/sys.html
    #If using command line arguments, be sure to add them to main function

    # Get file_path for dataset to analyze
    file_path = sys.argv[1]

    main(spark, file_path)    
