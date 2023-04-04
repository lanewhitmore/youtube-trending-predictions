import time
import random
import pandas as pd
from glob import glob as gl
import glob
import pprint
import argparse
import json
import subprocess
import sys
import os
import csv
#subprocess.check_call([sys.executable, "-m", "pip", "install", "gast==0.3.3"])
#subprocess.check_call([sys.executable, "-m", "pip", "install", "tensorboard==2.3.0"])
#subprocess.check_call([sys.executable, "-m", "pip", "install", "tenso"])
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'tensorflow==2.3.0'])
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'torch==1.4.0'])
subprocess.check_call([sys.executable, "-m", "pip", "install", "transformers==3.5.1"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "scikit-learn==0.23.1"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "matplotlib==3.2.1"])
#subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'sagemaker-tensorflow==2.1.0.1.0.0'])
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'smdebug==0.9.3'])

import tensorflow as tf
from tensorflow.keras.callbacks import CSVLogger
import smdebug.tensorflow as smd
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
import pandas as pd
import numpy as np
import boto3

from transformers import DistilBertTokenizer
from transformers import DistilBertConfig
from transformers import DistilBertModel
from transformers import TFDistilBertForSequenceClassification

from sagemaker_tensorflow import PipeModeDataset

from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, mean_absolute_error, r2_score

val_dir = "validation"
test_dir="test"
train_dir="train"


def select_data(record):
    x = {"title_input_ids": record["title_input_ids"], "title_input_mask": record["title_input_mask"],
            "tags_input_ids": record["tags_input_ids"], "title_input_mask": record["tags_input_mask"],
            "desc_input_ids": record["title_input_ids"], "title_input_mask": record["desc_input_mask"], 
             "segment_ids": record["segment_ids"]}
    return x
    

def tf_file_prep(input_filenames, steps_per_epoch, batch_size):
    print("Using input_filenames {}".format(input_filenames))
    name_features = {
        "title_input_ids": tf.io.FixedLenFeature([max_seq_length], tf.int64),
        "title_input_mask": tf.io.FixedLenFeature([max_seq_length], tf.int64),
        "tags_input_ids": tf.io.FixedLenFeature([max_seq_length], tf.int64),
        "tags_input_mask": tf.io.FixedLenFeature([max_seq_length], tf.int64),
        "desc_input_ids": tf.io.FixedLenFeature([max_seq_length], tf.int64),
        "desc_input_mask": tf.io.FixedLenFeature([max_seq_length], tf.int64),
        "segment_ids": tf.io.FixedLenFeature([max_seq_length], tf.int64),
    }
    dataset = tf.data.TFRecordDataset(input_filenames)
    dataset = dataset.repeat(epochs * steps_per_epoch * 100)
    
    
    def decode_record_to_tensorflow_ex(record, name_features):
        record = tf.io.parse_single_example(record, name_features)
        return record
    
    dataset = dataset.apply(
        tf.data.experimental.map_and_batch(
            lambda record: decode_record_to_tensorflow_ex(record, name_features),
            batch_size=batch_size,
            drop_remainder=False,
            num_parallel_calls=tf.data.experimental.AUTOTUNE)
    )
    
    return dataset
    



def import_train_predictors(input_filenames):
    records = tf_file_prep(input_filenames)
    x = {"title_input_ids": records["title_input_ids"], "title_input_mask": records["title_input_mask"],
            "tags_input_ids": records["tags_input_ids"], "title_input_mask": records["tags_input_mask"],
            "desc_input_ids": records["title_input_ids"], "title_input_mask": records["desc_input_mask"], 
             "segment_ids": records["segment_ids"]}
    x_train = records.map(x)
    return x_train
    
def import_train_response(train_dir):
    train_y = pd.read_csv(f"s3://sagemaker-us-east-1-492991381452/youtubeStatistics/splits/{train_dir}/train_view_count.csv")
    y_train = train_y.view_count
    print(y_train.shape)
    return y_train
    
def import_test_predictors(input_filenames):
    records = tf_file_prep(input_filenames)
    x = {"title_input_ids": records["title_input_ids"], "title_input_mask": records["title_input_mask"],
           "tags_input_ids": records["tags_input_ids"], "title_input_mask": records["tags_input_mask"],
           "desc_input_ids": records["title_input_ids"], "title_input_mask": records["desc_input_mask"], 
           "segment_ids": records["segment_ids"]}
    x_text = records.map(x)
    return x_test
    
def import_test_response(test_dir):
    test_y = pd.read_csv(f"s3://sagemaker-us-east-1-492991381452/youtubeStatistics/splits/{test_dir}/test_view_count.csv")
    y_test = test_y.view_count
    print(y_test.shape)
    return y_test

def import_validation_predictors(input_filenames):
    records = tf_file_prep(input_filenames)
    x = {"title_input_ids": records["title_input_ids"], "title_input_mask": records["title_input_mask"],
            "tags_input_ids": records["tags_input_ids"], "title_input_mask": records["tags_input_mask"],
            "desc_input_ids": records["title_input_ids"], "title_input_mask": records["desc_input_mask"], 
            "segment_ids": records["segment_ids"]}
    x_val = records.map(x)
    return x_val
    
def import_validation_response(val_dir):
    val_y = pd.read_csv(f"s3://sagemaker-us-east-1-492991381452/youtubeStatistics/splits/{val_dir}/val_view_count.csv")
    y_val = val_y.view_count
    print(y_val.shape)
    return y_val


    



def loading_checkpoint(checkpoint_path):
    glob_pattern = os.path.join(checkpoint_path, "*.h5")
    print("glob pattern {}".format(glob_pattern))
    
    list_of_checkpoints = glob.glob(glob_pattern)
    print("List of checkpoint files: {}".format(list_of_checkpoints))
    
    last_check = max(list_of_checkpoints)
    print("The latest checkpoint: {}".format(last_check))
    
    initial_epoch_num = last_check.rsplit("_", 1)[-1].split(".h5")[0]
    initial_epoch = int(initial_epoch_num)
    
    loaded_model = TFDistilBertModel.from_pretrained(last_check, config = config)
    
    print("loaded_model: {}".format(loaded_model))
    print("starting epoch from checkpoint: {}".format(initial_epoch))
    
    return loaded_model, initial_epoch




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--train_data", type=str, default=os.environ["SM_CHANNEL_TRAIN"])
    parser.add_argument("--validation_data", type=str, default=os.environ["SM_CHANNEL_VALIDATION"])
    parser.add_argument("--test_data", type=str, default=os.environ["SM_CHANNEL_TEST"])
    parser.add_argument("--output_dir", type=str, default=os.environ["SM_OUTPUT_DIR"])
    parser.add_argument("--hosts", type=list, default=json.loads(os.environ["SM_HOSTS"]))
    parser.add_argument("--current_host", type=str, default=os.environ["SM_CURRENT_HOST"])
    parser.add_argument("--num_gpus", type=int, default=os.environ["SM_NUM_GPUS"])
    parser.add_argument("--checkpoint_base_path", type=str, default="/opt/ml/checkpoints")
    parser.add_argument("--use_xla", type=eval, default=False)
    parser.add_argument("--use_amp", type=eval, default=False)
    parser.add_argument("--max_seq_length", type=int, default=64)
    parser.add_argument("--train_batch_size", type=int, default=128)
    parser.add_argument("--validation_batch_size", type=int, default=256)
    parser.add_argument("--test_batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=0.00003)
    parser.add_argument("--epsilon", type=float, default=0.00000001)
    parser.add_argument("--train_steps_per_epoch", type=int, default=None)
    parser.add_argument("--validation_steps", type=int, default=None)
    parser.add_argument("--test_steps", type=int, default=None)
    parser.add_argument("--freeze_bert_layer", type=eval, default=False)
    parser.add_argument("--enable_sagemaker_debugger", type=eval, default=False)
    parser.add_argument("--run_validation", type=eval, default=False)
    parser.add_argument("--run_test", type=eval, default=False)
    parser.add_argument("--run_sample_predictions", type=eval, default=False)
    parser.add_argument("--enable_tensorboard", type=eval, default=False)
    parser.add_argument("--enable_checkpointing", type=eval, default=False)
    parser.add_argument("--output_data_dir", type=str, default=os.environ["SM_OUTPUT_DATA_DIR"])
    
    
    args, _ = parser.parse_known_args()
    print("Args:")
    print(args)
    
    env_var = os.environ
    
    print("Environment Variables:")
    pprint.pprint(dict(env_var), width = 1)
    
    print("SageMaker Training Environment {}".format(env_var["SM_TRAINING_ENV"]))
    sm_training_env_json = json.loads(env_var["SM_TRAINING_ENV"])
    is_master = sm_training_env_json["is_master"]
    print("is_master {}".format(is_master))
    
    train_data = args.train_data
    print("train data: {}".format(train_data))
    
    validation_data = args.validation_data
    print("validation data: {}".format(validation_data))
    
    test_data = args.test_data
    print("test data: {}".format(test_data))
    
    model_directory = os.environ["SM_MODEL_DIR"]
    output_directory = args.output_dir
    print("output directory: {}".format(output_directory))
    
    hosts = args.hosts
    print("hosts: {}".format(hosts))
    
    current_host = args.current_host
    print("current host: {}".format(current_host))
    
    number_gpus = args.num_gpus
    print("number of gpus: {}".format(number_gpus))
    
    job_name = os.environ["SAGEMAKER_JOB_NAME"]
    print("job name: {}".format(job_name))
    
    use_xla = args.use_xla
    print("are you using xla? {}".format(use_xla))
    
    use_amp = args.use_amp
    print("are you using amp? {}".format(use_amp))
    
    max_seq_length = args.max_seq_length
    print("Max sequence length: {}".format(max_seq_length))
    
    train_batch_size = args.train_batch_size
    print("training batch size: {}".format(train_batch_size))
    
    validation_batch_size = args.validation_batch_size
    print("validation batch size: {}".format(validation_batch_size))
    
    test_batch_size = args.test_batch_size
    print("test batch size: {}".format(test_batch_size))
    
    epochs = args.epochs 
    print("epochs: {}".format(epochs))
    
    learning_rate = args.learning_rate
    print("learning rate: {}".format(learning_rate))
    
    epsilon = args.epsilon
    print("epsilon: {}".format(epsilon))
    
    train_steps_per_epoch = args.train_steps_per_epoch
    print("train steps per epoch: {}".format(train_steps_per_epoch))
    
    validation_steps = args.validation_steps
    print("validation steps: {}".format(validation_steps))
    
    test_steps = args.test_steps
    print("test steps: {}".format(test_steps))
    
    freeze_bert_layer = args.freeze_bert_layer
    print("freeze bert layer: {}".format(freeze_bert_layer))
    
    enable_sagemaker_debugger = args.enable_sagemaker_debugger
    print("enable sagemaker debugger: {}".format(enable_sagemaker_debugger))
    
    run_validation = args.run_validation
    print("run validation: {}".format(run_validation))
    
    run_test = args.run_test
    print("run test: {}".format(run_test))
    
    run_sample_predictions = args.run_sample_predictions
    print("run sample predictions: {}".format(run_sample_predictions))
    
    enable_tensorboard = args.enable_tensorboard
    print("enable tensorboard: {}".format(enable_tensorboard))
    
    enable_checkpointing = args.enable_checkpointing
    print("enable checkpointing: {}".format(enable_checkpointing))
    
    if is_master:
        checkpoint_path = args.checkpoint_base_path
    else:
        checkpoint_path = "/tmp/checkpoints"
    print("checkpoint path: {}".format(checkpoint_path))
    
    
    pipe_mode_string = os.environ.get("SM_INPUT_DATA_CONFIG", "")
    pipe_mode = pipe_mode_string.find("Pipe") >= 0
    print("using the pipe mode: {}".format(pipe_mode))
    
    transformer_tuned_model_path = os.path.join(model_directory, "transformers/tuned/")
    os.makedirs(transformer_tuned_model_path, exist_ok = True)
    
    
    tf_saved_model_path = os.path.join(model_directory, "tensorflow/saved_model/mod")
    os.makedirs(tf_saved_model_path, exist_ok = True)



if __name__ == "__main__":
    
    print("Building X and y train....")
    train_data_filenames = gl(os.path.join(train_data, "*.tfrecord"))
    X_train = tf_file_prep(train_data_filenames, train_steps_per_epoch, train_batch_size).map(select_data)
    
    
    y_train = import_train_response(train_dir)
    print(y_train.head())
    
    
    
    print("Building X and y test....")
    test_data_filenames = gl(os.path.join(test_data, "*.tfrecord"))
    X_test = tf_file_prep(test_data_filenames, test_steps, test_batch_size).map(select_data)
    
    
    y_test = import_test_response(test_dir)
    
    
    print("Building X and y validation....")
    validation_data_filenames = gl(os.path.join(validation_data, "*.tfrecord"))
    X_val = tf_file_prep(validation_data_filenames, validation_steps, validation_batch_size).map(select_data)
    
    y_test = import_validation_response(val_dir)
    
    
    maxlen=300
    vocab_size = 30522
    
    print("Constructing Keras LSTM Model....")
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
                
    #transformer_model = TFDistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels = 1)
    
    title_ids = tf.keras.layers.Input(shape=(max_seq_length,), name="title_input_ids", dtype = "int32")
    title_mask = tf.keras.layers.Input(shape=(max_seq_length,),name = "title_input_mask", dtype = "int32")
    title_embedding_layer = transformer_model.distilbert(title_ids, attention_mask = title_mask)[0]
    print("TITLE EMBEDDING LAYER:", title_embedding_layer)
    
    #optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate)
    # creating lstm model
    #model = tf.keras.models.Sequential()
    #model.add(tf.keras.embedding.Embedding(max_features, title_embedding_layer, input_length=maxlen, weights=[embedding_matrix]))
   # model.add(tf.keras.layers.Dropout(0.2))
    #model.add(tf.keras.layers.Bidirectional(tf.keras.layers.CuDNNLSTM(64, return_sequences=True)))
    #model.add(tf.keras.layers.Dropout(0.2))
   # model.add(tf.keras.layers.Bidirectional(tf.keras.layers.CuDNNLSTM(32)))
    #model.add(tf.keras.layers.Dropout(0.2))
   # model.add(tf.keras.layers.Dense(1))
    # Compile model
    #model.compile(loss='mse', optimizer=optimizer, metrics=['mse', 'mae','mape'])                              
                                
 
    print("Training Model....")
    
    
    data_location = "/opt/ml/model/training.log"
    print("Model Logging Location: {}".format(data_location))
    csv_logger = CSVLogger(data_location)
    
    
    #model.fit(x=[X_train.title_inputs_ids,X_train.title_input_mask], y=y_train, batch_size=train_batch_size, epochs=epochs, verbose=1, callbacks=[csv_logger])
    
    print("Predicting validation set....")
    
    #val_pred = model.predict([X_val.title_input_ids, X_val.title_input_mask], y_val)
    
    #submission = pd.DataFrame(y_val)
    #submission.head()

    #submission['prediction'] = val_pred
    #submission.head()

    #data_location = "/opt/ml/model/submission.csv"

    #submission.to_csv(data_location, index=False)
    
    print("Predicting test set.....")
    
    
    print("Finished.....!")
    
    
    
    
    
    
    
    

