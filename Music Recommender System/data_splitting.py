#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys

from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.window import Window


def main(spark, file_path):
    '''Main routine for run for Storage optimization template.
    Parameters
    ----------
    spark : SparkSession object
    '''
    # Reading the whole dataset
    df = spark.read.parquet(file_path)

    # Removing songs with few unique listeners
    df = df.drop("row_number").drop("grouped_user").drop("unique users").drop("total_count").drop("train_samples").drop("validation_samples")
    grouped_df = df.groupBy("recording_msid").agg(F.countDistinct("user_id").alias("unique users"))
    value_1 = 2
    filtered_grouped_df = grouped_df.filter(grouped_df["unique users"] >= value)
    filtered_grouped_df = filtered_grouped_df.withColumnRenamed("recording_msid", "filtered_recording_msid")
    final_df = df.join(filtered_grouped_df, df["recording_msid"] == filtered_grouped_df["filtered_recording_msid"])
    final_df = final_df.drop("filtered_recording_msid").drop("grouped_user").drop("unique_users").drop("unique user")

    # Removing users who listen to very few tracks

    grouped_df_2 = final_df.groupBy("user_id").agg(F.countDistinct("recording_msid").alias("number of unique tracks heard"))
    value_2 = 10
    filtered_grouped_df_2 = grouped_df_2.filter(grouped_df_2["number of unique tracks heard"] >= value_2)
    filtered_grouped_df_2 = filtered_grouped_df_2.withColumnRenamed("user_id", "filtered_user_id")
    final_df_2 = final_df.join(filtered_grouped_df_2, final_df["user_id"] == filtered_grouped_df_2["filtered_user_id"])
    final_df_2 = final_df_2.drop("filtered_user_id").drop("unique users").drop("number of unique tracks heard")
    
    # Creating row numbers for each user's listening history
    windowSpec = Window.partitionBy("user_id").orderBy("user_id", "recording_msid")
    df = final_df_2.withColumn("row_number", F.row_number().over(windowSpec))
    
    # Splitting 80% of each user's interactions into training set and 20% into validation set
    grouped_df = df.groupBy("user_id").agg(F.count("*").alias("total_count"))
    grouped_df = grouped_df.withColumn("train_samples", F.floor(grouped_df["total_count"]*0.8))
    grouped_df = grouped_df.withColumn("validation_samples", grouped_df["total_count"] - grouped_df["train_samples"])
    grouped_df = grouped_df.withColumnRenamed("user_id", "grouped_user")
    train_df = df.join(grouped_df, (df["user_id"] == grouped_df["grouped_user"]) & (df["row_number"] <= grouped_df["train_samples"]))
    validation_df = df.join(grouped_df, (df["user_id"] == grouped_df["grouped_user"]) & (df["row_number"] > grouped_df["train_samples"]))
    
    # Save training and validation sets as parquet files
    train_df.write.parquet("preprocessed_training_small.parquet", mode = "overwrite")
    validation_df.write.parquet("preprocessed_validation_small.parquet", mode = "overwrite")

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
    
