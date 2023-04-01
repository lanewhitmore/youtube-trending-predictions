# Imports
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import functools
import multiprocessing

from datetime import datetime
from time import gmtime, strftime, sleep

import sys
import re
import collections
import argparse
import json
import os
import csv
import glob
from pathlib import Path
import time
import boto3
import subprocess


subprocess.check_call([sys.executable, "-m", "conda", "install", "-c", "conda-forge", "transformers==3.5.1", "-y"])
from transformers import DistilBertTokenizer
from transformers import DistilBertConfig

subprocess.check_call([sys.executable, "-m", "conda", "install", "-c", "anaconda", "tensorflow==2.3.0", "-y"])
import tensorflow as tf
from tensorflow import keras

subprocess.check_call([sys.executable, "-m", "pip", "install", "matplotlib==3.2.1"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "sagemaker==2.24.1"])
import pandas as pd
import re
import sagemaker
from sagemaker.session import Session
from sagemaker.feature_store.feature_group import FeatureGroup
from sagemaker.feature_store.feature_definition import (
    FeatureDefinition,
    FeatureTypeEnum,
)

# saving role information for access to aws sagemaker and buckets with access to make changes

region = os.environ["AWS_DEFAULT_REGION"]
print("Region: {}".format(region))

sts = boto3.Session(region_name=region).client(service_name="sts", region_name=region)
caller_identity = sts.get_caller_identity()
print("caller_identity: {}".format(caller_identity))

assumed_role_arn = caller_identity["Arn"]
print("(assumed_role) caller_identity_arn: {}".format(assumed_role_arn))

assumed_role_name = assumed_role_arn.split("/")[-2]

iam = boto3.Session(region_name=region).client(service_name="iam", region_name=region)
get_role_response = iam.get_role(RoleName=assumed_role_name)
print("get_role_response {}".format(get_role_response))
role = get_role_response["Role"]["Arn"]
print("role {}".format(role))

bucket = sagemaker.Session().default_bucket()
print("The DEFAULT BUCKET is {}".format(bucket))


# connection to sagemaker session and the s3 session using the information obtained from above
sm = boto3.Session(region_name=region).client(service_name="sagemaker", region_name=region)

featurestore_runtime = boto3.Session(region_name=region).client(
    service_name="sagemaker-featurestore-runtime", region_name=region
)

s3 = boto3.Session(region_name=region).client(service_name="s3", region_name=region)

sagemaker_session = sagemaker.Session(
    boto_session=boto3.Session(region_name=region),
    sagemaker_client=sm,
    sagemaker_featurestore_runtime_client=featurestore_runtime,
)



# global variables
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

TITLE_COLUMN = "title"
TAGS_COLUMN = "tags"
DESCRIPTION_COLUMN = "description"
VIDEO_ID_COLUMN = "video_id"
VIEW_COUNT_COLUMN = "view_count"



def df_obj_to_string(data_frame):
    # converting the object types in the dataframe to strings to avoid issues with the distilbert package
    for label in data_frame.columns:
        if data_frame.dtypes[label] == "object":
            data_frame[label] = data_frame[label].astype("str").astype("string")
    return data_frame

def feature_group_complete_and_log(feature_group):
    # establishing logging information for the creation of the feature group
    try:
        status = feature_group.describe().get("FeatureGroupStatus")
        print("Feature Group status: {}".format(status))
        while status == "Creating":
            print("Waiting for Feature Group Creation")
            time.sleep(5)
            status = feature_group.describe().get("FeatureGroupStatus")
            print("Feature Group status: {}".format(status))
        if status != "Created":
            print("Feature Group status: {}".format(status))
            raise RuntimeError(f"Failed to create feature group {feature_group.name}")
        print(f"FeatureGroup {feature_group.name} successfully created.")
    except:
        print("No feature group created yet.")


def create_or_load_feature_group(prefix, feature_group_name):

    # Feature Definitions the same as in the ipynb
    feature_definitions = [
        FeatureDefinition(feature_name="title_input_ids", feature_type=FeatureTypeEnum.STRING),
        FeatureDefinition(feature_name="title_input_mask", feature_type=FeatureTypeEnum.STRING),
        FeatureDefinition(feature_name="tags_input_ids", feature_type=FeatureTypeEnum.STRING),
        FeatureDefinition(feature_name="tags_input_mask", feature_type=FeatureTypeEnum.STRING),
        FeatureDefinition(feature_name="desc_input_ids", feature_type=FeatureTypeEnum.STRING),
        FeatureDefinition(feature_name="desc_input_mask", feature_type=FeatureTypeEnum.STRING),
        FeatureDefinition(feature_name="segment_ids", feature_type=FeatureTypeEnum.STRING),
        FeatureDefinition(feature_name="date", feature_type=FeatureTypeEnum.STRING),
        FeatureDefinition(feature_name="video_id", feature_type=FeatureTypeEnum.STRING),
        FeatureDefinition(feature_name="split_type", feature_type=FeatureTypeEnum.STRING),
        FeatureDefinition(feature_name="view_count", feature_type=FeatureTypeEnum.INTEGRAL),
    ]

    feature_group = FeatureGroup(
        name=feature_group_name, feature_definitions=feature_definitions, sagemaker_session=sagemaker_session
    )

    print("Feature Group: {}".format(feature_group))

    try:
        print(
            "Waiting for existing Feature Group to become available if it is being created by another instance in our cluster..."
        )
        feature_group_complete_and_log(feature_group)
    except Exception as e:
        print("Before CREATE FG wait exeption: {}".format(e))
    #        pass

    try:
        record_identifier_feature_name = "video_id"
        event_time_feature_name = "date"

        print("Creating Feature Group with role {}...".format(role))
        feature_group.create(
            s3_uri=f"s3://{bucket}/{prefix}",
            record_identifier_name=record_identifier_feature_name,
            event_time_feature_name=event_time_feature_name,
            role_arn=role,
            enable_online_store=False,
        )
        print("Creating Feature Group. Completed.")

        print("Waiting for new Feature Group to become available...")
        feature_group_complete_and_log(feature_group)
        print("Feature Group available.")
        feature_group.describe()

    except Exception as e:
        print("Exception: {}".format(e))

    return feature_group


class InputFeatures(object):
    # the input features for the Bert vectorization/tokenization

    def __init__(self, title_input_ids, title_input_mask, tags_input_ids, tags_input_mask, desc_input_ids, desc_input_mask, segment_ids, video_id, date, view_count):
        self.title_input_ids = title_input_ids
        self.title_input_mask = title_input_mask
        self.tags_input_ids = tags_input_ids
        self.tags_input_mask = tags_input_mask
        self.desc_input_ids = desc_input_ids
        self.desc_input_mask = desc_input_mask
        self.segment_ids = segment_ids
        self.video_id = video_id
        self.date = date
        self.view_count = view_count



class Input(object):
    # input establishes the features to be converted
    def __init__(self, title, tags, description, video_id, date, view_count):
        self.title = str(title)
        self.tags = str(tags)
        self.description = str(description)
        self.video_id = str(video_id)
        self.date = date
        self.view_count = view_count


def convert_input(the_input, max_seq_length):
    title_tokens = tokenizer.tokenize(the_input.title)
    encoded_title = tokenizer.encode_plus(the_input.title,
                                          pad_to_max_length=True,
                                          max_length=max_seq_length,
                                          truncation=True)
    
    title_input_ids = encoded_title['input_ids']
    title_input_mask = encoded_title['attention_mask']
    
    tags_tokens = tokenizer.tokenize(the_input.tags)
    encoded_tags = tokenizer.encode_plus(the_input.tags,
                                         pad_to_max_length=True,
                                         max_length=max_seq_length,
                                         truncation=True)
    tags_input_ids = encoded_tags['input_ids']
    tags_input_mask = encoded_tags['attention_mask']

    description_tokens = tokenizer.tokenize(the_input.description)
    encoded_desc = tokenizer.encode_plus(the_input.description,
                                         pad_to_max_length=True,
                                         max_length=max_seq_length,
                                         truncation=True)
    desc_input_ids = encoded_desc['input_ids']
    desc_input_mask = encoded_desc['attention_mask']

    segment_ids = [0] * max_seq_length

    features = InputFeatures(
        title_input_ids=title_input_ids,
        title_input_mask=title_input_mask,
        tags_input_ids=tags_input_ids,
        tags_input_mask=tags_input_mask,
        desc_input_ids=desc_input_ids,
        desc_input_mask=desc_input_mask,
        segment_ids=segment_ids,
        video_id=the_input.video_id,
        date=the_input.date,
        view_count=the_input.view_count,
    )

    return features


def transform_inputs_to_tfrecord(inputs, output_file, max_seq_length):

    records = []

    tf_record_writer = tf.io.TFRecordWriter(output_file)

    for (input_idx, the_input) in enumerate(inputs):
        if input_idx % 10000 == 0:
            print("Writing input {} of {}\n".format(input_idx, len(inputs)))

        features = convert_input(the_input, max_seq_length)

        all_features = collections.OrderedDict()
        all_features["title_input_ids"] = tf.train.Feature(int64_list=tf.train.Int64List(value=features.title_input_ids))
        all_features["title_input_mask"] = tf.train.Feature(int64_list=tf.train.Int64List(value=features.title_input_mask))
        all_features["tags_input_ids"] = tf.train.Feature(int64_list=tf.train.Int64List(value=features.tags_input_ids))
        all_features["tags_input_mask"] = tf.train.Feature(int64_list=tf.train.Int64List(value=features.tags_input_mask))
        all_features["desc_input_ids"] = tf.train.Feature(int64_list=tf.train.Int64List(value=features.desc_input_ids))
        all_features["desc_input_mask"] = tf.train.Feature(int64_list=tf.train.Int64List(value=features.desc_input_mask))
        all_features["segment_ids"] = tf.train.Feature(int64_list=tf.train.Int64List(value=features.segment_ids))

        tf_record = tf.train.Example(features=tf.train.Features(feature=all_features))
        tf_record_writer.write(tf_record.SerializeToString())

        records.append(
            {  
                "title_input_ids": features.title_input_ids,
                "title_input_mask": features.title_input_mask,
                "tags_input_ids": features.tags_input_ids,
                "tags_input_mask": features.tags_input_mask,
                "desc_input_ids": features.desc_input_ids,
                "desc_input_mask": features.desc_input_mask,
                "segment_ids": features.segment_ids,
                "video_id": the_input.video_id,
                "date": the_input.date,
                "view_count": the_input.view_count,
            }
        )

    tf_record_writer.close()

    return records


def list_arg(raw_value):
    """argparse type for a list of strings"""
    return str(raw_value).split(",")


def parse_args():
    # parsing the configuration file stored in .json
    resconfig = {}
    try:
        with open("/opt/ml/config/resourceconfig.json", "r") as cfgfile:
            resconfig = json.load(cfgfile)
    except FileNotFoundError:
        print("/opt/ml/config/resourceconfig.json not found.  current_host is unknown.")
        pass  # Ignore

    # Local testing with CLI args
    parser = argparse.ArgumentParser(description="Process")

    parser.add_argument(
        "--hosts",
        type=list_arg,
        default=resconfig.get("hosts", ["unknown"]),
    )
    parser.add_argument(
        "--current-host",
        type=str,
        default=resconfig.get("current_host", "unknown"),
    )
    parser.add_argument(
        "--input-data",
        type=str,
        default="/opt/ml/processing/input/data",
    )
    parser.add_argument(
        "--output-data",
        type=str,
        default="/opt/ml/processing/output",
    )
    parser.add_argument(
        "--train-split-percentage",
        type=float,
        default=0.85,
    )
    parser.add_argument(
        "--validation-split-percentage",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--test-split-percentage",
        type=float,
        default=0.10,
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--feature-store-offline-prefix",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--feature-group-name",
        type=str,
        default=None,
    )

    return parser.parse_args()


def _transform_csv_to_tfrecord(file, max_seq_length, prefix, feature_group_name):
    print("file {}".format(file))
    print("max_seq_length {}".format(max_seq_length))
    print("prefix {}".format(prefix))
    print("feature_group_name {}".format(feature_group_name))

    # reloading feature group
    feature_group = create_or_load_feature_group(prefix, feature_group_name)

    filename_without_extension = Path(Path(file).stem).stem
    
    # reading the csv file gaming
    df = pd.read_csv(file)
    
    # shape of data frame print
    print("Shape of dataframe {}".format(df.shape))
    # showing the percentage of split for train/test/val
    print("train split percentage {}".format(args.train_split_percentage))
    print("validation split percentage {}".format(args.validation_split_percentage))
    print("test split percentage {}".format(args.test_split_percentage))
    
    # assigning the holdout for test/val from subtracting the train % which is 85%
    holdout_percentage = 1.00 - args.train_split_percentage
    print("holdout percentage {}".format(holdout_percentage))
    
    # splitting the df train set
    df_train, df_holdout = train_test_split(df, test_size=holdout_percentage)
    
    # calculating the test holdout before splitting validation
    test_holdout_percentage = args.test_split_percentage / holdout_percentage
    
    print("test holdout percentage {}".format(test_holdout_percentage))
    
    # splitting the validation and test set
    df_validation, df_test = train_test_split(df_holdout, test_size=test_holdout_percentage)
    
    # reseting the index of each new df
    df_train = df_train.reset_index(drop=True)
    df_validation = df_validation.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)
    
            # confirm the final shapes of the dataframe are correct 85/10/5
    print("Shape of train dataframe {}".format(df_train.shape))
    print("Shape of validation dataframe {}".format(df_validation.shape))
    print("Shape of test dataframe {}".format(df_test.shape))

    # timestamp for the date variable
    timestamp = datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")
    print(timestamp)
    # finally the dataframes are assigned to create the inputs
    train_inputs = df_train.apply(
        lambda x: Input(
            view_count=x[VIEW_COUNT_COLUMN], title=x[TITLE_COLUMN], tags=x[TAGS_COLUMN], description=x[DESCRIPTION_COLUMN], video_id=x[VIDEO_ID_COLUMN], date=timestamp
        ),
        axis=1,
    )

    validation_inputs = df_validation.apply(
        lambda x: Input(
            view_count=x[VIEW_COUNT_COLUMN], title=x[TITLE_COLUMN], tags=x[TAGS_COLUMN], description=x[DESCRIPTION_COLUMN], video_id=x[VIDEO_ID_COLUMN], date=timestamp
        ),
        axis=1,
    )

    test_inputs = df_test.apply(
        lambda x: Input(
            view_count=x[VIEW_COUNT_COLUMN], title=x[TITLE_COLUMN], tags=x[TAGS_COLUMN], description=x[DESCRIPTION_COLUMN], video_id=x[VIDEO_ID_COLUMN], date=timestamp
        ),
        axis=1,
    )

   # saving the data within the s3 bucker
    train_data = "{}/bert/train".format(args.output_data)
    print(train_data)
    validation_data = "{}/bert/validation".format(args.output_data)
    test_data = "{}/bert/test".format(args.output_data)
    
    train_records = transform_inputs_to_tfrecord(
        train_inputs,
        "{}/part-{}-{}.tfrecord".format(train_data, args.current_host, filename_without_extension),
        max_seq_length,
    )

    validation_records = transform_inputs_to_tfrecord(
        validation_inputs,
        "{}/part-{}-{}.tfrecord".format(validation_data, args.current_host, filename_without_extension),
        max_seq_length,
    )

    test_records = transform_inputs_to_tfrecord(
        test_inputs,
        "{}/part-{}-{}.tfrecord".format(test_data, args.current_host, filename_without_extension),
        max_seq_length,
    )

    df_train_records = pd.DataFrame.from_dict(train_records)
    df_train_records["split_type"] = "train"
    df_train_records.head()

    df_validation_records = pd.DataFrame.from_dict(validation_records)
    df_validation_records["split_type"] = "validation"
    df_validation_records.head()

    df_test_records = pd.DataFrame.from_dict(test_records)
    df_test_records["split_type"] = "test"
    df_test_records.head()

    
    df_fs_train_records = df_obj_to_string(df_train_records)
    df_fs_validation_records = df_obj_to_string(df_validation_records)
    df_fs_test_records = df_obj_to_string(df_test_records)
# Add record to feature store
    print("Ingesting Features...")
    feature_group.ingest(data_frame=df_fs_train_records, max_workers=3, wait=True)
    feature_group.ingest(data_frame=df_fs_validation_records, max_workers=3, wait=True)
    feature_group.ingest(data_frame=df_fs_test_records, max_workers=3, wait=True)
    
    offline_store_status = None
    while offline_store_status != 'Active':
        try:
            offline_store_status = feature_group.describe()['OfflineStoreStatus']['Status']
        except:
            pass
        print('Offline store status: {}'.format(offline_store_status))    
    print('...features ingested!')


def process(args):
    print("Current host: {}".format(args.current_host))

    feature_group = create_or_load_feature_group(prefix=args.feature_store_offline_prefix, 
                                                 feature_group_name=args.feature_group_name)

    feature_group.describe()

    print(feature_group.as_hive_ddl())

    train_data = "{}/bert/train".format(args.output_data)
    validation_data = "{}/bert/validation".format(args.output_data)
    test_data = "{}/bert/test".format(args.output_data)

    transform_csv_to_tfrecord = functools.partial(
        _transform_csv_to_tfrecord,
        max_seq_length=args.max_seq_length,
        prefix=args.feature_store_offline_prefix,
        feature_group_name=args.feature_group_name,
    )

    input_files = glob.glob("{}/*.csv".format(args.input_data))

    num_cpus = multiprocessing.cpu_count()
    print("num_cpus {}".format(num_cpus))

    p = multiprocessing.Pool(num_cpus)
    p.map(transform_csv_to_tfrecord, input_files)

    print("Listing contents of {}".format(args.output_data))
    dirs_output = os.listdir(args.output_data)
    for file in dirs_output:
        print(file)

    print("Listing contents of {}".format(train_data))
    dirs_output = os.listdir(train_data)
    for file in dirs_output:
        print(file)

    print("Listing contents of {}".format(validation_data))
    dirs_output = os.listdir(validation_data)
    for file in dirs_output:
        print(file)

    print("Listing contents of {}".format(test_data))
    dirs_output = os.listdir(test_data)
    for file in dirs_output:
        print(file)

    offline_store_contents = None
    while offline_store_contents is None:
        objects_in_bucket = s3.list_objects(Bucket=bucket, Prefix=args.feature_store_offline_prefix)
        if "Contents" in objects_in_bucket and len(objects_in_bucket["Contents"]) > 1:
            offline_store_contents = objects_in_bucket["Contents"]
        else:
            print("Waiting for data in offline store...\n")
            sleep(60)

    print("Data available.")

    print("Complete")


if __name__ == "__main__":
    args = parse_args()
    print("Loaded arguments:")
    print(args)

    print("Environment variables:")
    print(os.environ)

    process(args)