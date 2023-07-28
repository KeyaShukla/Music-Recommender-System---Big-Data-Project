#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import pyspark.sql.functions as F
from pyspark.sql import SparkSession
from pyspark.sql.window import Window
from pyspark.mllib.evaluation import RankingMetrics


def calculatePopularity(df, beta):
    df = df.groupBy('user_id', 'recording_msid').agg(F.count('*').alias('song_listens'))
    df = df.groupBy('recording_msid').agg(F.sum('song_listens').alias('total_interactions'),\
                                          F.count('song_listens').alias('unique_users'))
    df = df.withColumn('popularity', df['total_interactions'] / (df['unique_users'] + beta))
    return df

def main(spark, file_path):
    '''Main routine for run for Storage optimization template.
    Parameters
    ----------
    spark : SparkSession object
    '''
    
    betas = [0.01, 0.05, 0.1, 0.5, 1, 10, 100, 1000, 5000, 10000]
    betas = [5000] # This array has the optimal value of beta that will be used to run on the test data
    predictions = 100

    train_data = spark.read.parquet(file_path + '/training_small.parquet')
    #test_data = spark.read.parquet(file_path + '/validation_small.parquet').select('user_id', 'recording_msid')
    test_data = spark.read.parquet('/user/bm106_nyu_edu/1004-project-2023/interactions_test.parquet')

    ap_values = []

    for beta in betas:
        data = calculatePopularity(train_data, beta)
        
        recs = data.select('recording_msid', 'popularity').distinct()\
                   .sort(F.desc(F.col('popularity')), F.col('recording_msid')).limit(predictions)

        recs = recs.select('recording_msid').agg(F.collect_set('recording_msid').alias('pred'))
        test = test_data.groupBy('user_id').agg(F.collect_set('recording_msid').alias('true'))
    
        eval_df = recs.crossJoin(test)

        eval_rdd = eval_df.rdd.map(lambda row: (row['pred'], row['true']))
        metrics = RankingMetrics(eval_rdd)
        
        map_val = metrics.meanAveragePrecisionAt(predictions)
        ap = metrics.precisionAt(predictions)
        ap_values.append(ap)
        print("Beta: %f, MAP: %f, AP: %f" % (beta, map_val, ap))

    print("Best AP: %f" % max(ap_values))
    print("Best beta: %F" % betas[ap_values.index(max(ap_values))])

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
    
