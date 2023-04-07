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
subprocess.check_call([sys.executable, "-m", "pip", "install", "transformers==3.5.1"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "scikit-learn==0.23.1"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "matplotlib==3.2.1"])
#subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'sagemaker-tensorflow==2.1.0.1.0.0'])
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'smdebug==0.9.3'])

import tensorflow as tf
import smdebug.tensorflow as smd
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
import pandas as pd
import numpy as np


from transformers import DistilBertTokenizer
from transformers import DistilBertConfig
from transformers import DistilBertModel
from transformers import TFDistilBertForSequenceClassification

from sagemaker_tensorflow import PipeModeDataset

from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, mean_absolute_error, r2_score


def select_data_and_view_count(record):
    x = {"title_input_ids": record["title_input_ids"], "title_input_mask": record["title_input_mask"],
           "tags_input_ids": record["tags_input_ids"], "title_input_mask": record["tags_input_mask"],
           "desc_input_ids": record["title_input_ids"], "title_input_mask": record["desc_input_mask"], 
           "segment_ids": record["segment_ids"]}

    y = record["view_count"]
    
    return (x, y)

def file_dataset_builder(channel, input_filenames, pipe_mode, is_training, drop_remainder, batch_size, epochs, steps_per_epoch, max_seq_length):
    if pipe_mode:
        print("Using pipe_mode with channel {}".format(channel))
        dataset = PipeModeDataset(channel = channel, record_format = "TFRecord")
    else:
        print("Using input_filenames {}".format(input_filenames))
        dataset = tf.data.TFRecordDataset(input_filenames)
    
    dataset = dataset.repeat(epochs * steps_per_epoch * 100)
    
    name_features = {
        "title_input_ids": tf.io.FixedLenFeature([max_seq_length], tf.int64),
        "title_input_mask": tf.io.FixedLenFeature([max_seq_length], tf.int64),
        "tags_input_ids": tf.io.FixedLenFeature([max_seq_length], tf.int64),
        "tags_input_mask": tf.io.FixedLenFeature([max_seq_length], tf.int64),
        "desc_input_ids": tf.io.FixedLenFeature([max_seq_length], tf.int64),
        "desc_input_mask": tf.io.FixedLenFeature([max_seq_length], tf.int64),
        "segment_ids": tf.io.FixedLenFeature([max_seq_length], tf.int64),
        "view_count": tf.io.FixedLenFeature([], tf.int64),
    }
    
    def decode_record_to_tensorflow_ex(record, name_features):
        record = tf.io.parse_single_example(record, name_features)
        return record
    
    dataset = dataset.apply(
        tf.data.experimental.map_and_batch(
            lambda record: decode_record_to_tensorflow_ex(record, name_features),
            batch_size=batch_size,
            drop_remainder=drop_remainder,
            num_parallel_calls=tf.data.experimental.AUTOTUNE)
    )
    
    dataset = dataset.shuffle(buffer_size=1000, reshuffle_each_iteration=True)
    
    row_count = 0
    
    print("--------- {} -----------".format(channel))
    for row in dataset.as_numpy_iterator():
        print(row)
        if row_count ==5:
            break
        row_count = row_count +1
        
    return dataset

def loading_checkpoint(checkpoint_path):
    glob_pattern = os.path.join(checkpoint_path, "*.h5")
    print("glob pattern {}".format(glob_pattern))
    
    list_of_checkpoints =gl(glob_pattern)
    print("List of checkpoint files: {}".format(list_of_checkpoints))
    
    last_check = max(list_of_checkpoints)
    print("The latest checkpoint: {}".format(last_check))
    
    initial_epoch_num = last_check.rsplit("_", 1)[-1].split(".h5")[0]
    initial_epoch = int(initial_epoch_num)
    
    loaded_model = TFDistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", config=config, num_labels = 1)
    
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
    
    tensorboard_logs_path = os.path.join(model_directory, "tensorboard/")
    os.makedirs(tensorboard_logs_path, exist_ok=True)
    
    
    distribute_strategy = tf.distribute.MirroredStrategy()
    
    with distribute_strategy.scope():
        tf.config.optimizer.set_jit(use_xla)
        tf.config.optimizer.set_experimental_options({"auto mixed precision": use_amp})
        
        train_data_filenames = gl(os.path.join(train_data, "*.tfrecord"))
        print("train_data_filenames {}".format(train_data_filenames))
        train_dataset = file_dataset_builder(
            channel = "train",
            input_filenames = train_data_filenames,
            pipe_mode = pipe_mode,
            is_training = True,
            drop_remainder = False,
            batch_size = train_batch_size,
            epochs = epochs,
            steps_per_epoch = train_steps_per_epoch,
            max_seq_length = max_seq_length).map(select_data_and_view_count)
        
        tokenizer = None
        config = None 
        model = None
        transformer_model = None
        
        successful_download = False
        retries = 0
        while retries < 5 and not successful_download:
            try:
                tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
                
                transformer_model = TFDistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", config=config, num_labels = 1)
                
                title_ids = tf.keras.layers.Input(shape=(max_seq_length,), name="title_input_ids", dtype = "int32")
                title_mask = tf.keras.layers.Input(shape=(max_seq_length,),name = "title_input_mask", dtype = "int32")
             #   tags_ids = tf.keras.layers.Input(shape=(max_seq_length,), name="tags_input_ids", dtype = "int32")
                #tags_mask = tf.keras.layers.Input(shape=(max_seq_length,),name = "tags_input_mask", dtype = "int32")
               # desc_ids = tf.keras.layers.Input(shape=(max_seq_length,), name="desc_input_ids", dtype = "int32")
               # desc_mask = tf.keras.layers.Input(shape=(max_seq_length,),name = "desc_input_mask", dtype = "int32")
                
                title_embedding_layer = transformer_model.distilbert(title_ids, attention_mask = title_mask)[0]
                #tags_embedding_layer = transformer_model.distilbert(tags_ids, attention_mask = tags_mask)[0]
                #desc_embedding_layer = transformer_model.distilbert(desc_ids, attention_mask = desc_mask)[0]
                
                ## model to build after getting initial model to work
               # X = tf.keras.models.Sequential(
                           # tf.keras.layers.Bidirectional(
                               # tf.keras.layers.LSTM(64, return_sequences = True, dropout = 0.1, recurrent_dropout = 0.1)
                            #)
                        #)(title_embedding_layer)
               # X = tf.keras.layersBidirectional(tf.keras.layers.LSTM(32, dropout=0.5))
              #  X = tf.keras.layers.Dense(16, activation="relu")(X)
               # X = tf.keras.layers.Dropout(0.5)(X)
               # X = tf.keras.layers.Dense(1)(X)
                
                # trying to get this initial model to run then a more complex one will be implemented here
                X = tf.keras.layers.Bidirectional(
                    tf.keras.layers.LSTM(50, return_sequences=True, dropout=0.1, recurrent_dropout=0.1)
                )(embedding_layer)
                X = tf.keras.layers.GlobalMaxPool1D()(X)
                X = tf.keras.layers.Dense(50, activation="relu")(X)
                X = tf.keras.layers.Dropout(0.2)(X)
                X = tf.keras.layers.Dense(1, activation='linear')(X)
                
                model = tf.keras.Model(inputs=[title_ids, title_mask], outputs=X)
                
                for layer in model.layers[:3]:
                    layer.trainable = not freeze_bert_layer

                successful_download = True
                print("Sucessfully downloaded after {} retries.".format(retries))
            except:
                retries = retries + 1
                random_sleep = random.randint(1, 30)
                print("Retry #{}.  Sleeping for {} seconds".format(retries, random_sleep))
                time.sleep(random_sleep)

        callbacks = []
        
        initial_epoch = 0
        
        if enable_checkpointing:
            print("------ Checkpoint Enabled -------")
            
            os.makedirs(checkpoint_path, exist_ok=True)
            if os.listdir(checkpoint_path):
                print("------ Found Checkpoint -------")
                print(checkpoint_path)
                model, initial_epoch = loading_checkpoint(checkpoint_path)
                print("----- Using checkpoint model {} -----".format(model))
                
                
            checkpoint_callback = ModelCheckpoint(filepath = os.path.join(checkpoint_path, "tf_model_{epoch:05d}.h5"),
                                                                                   save_weights_only = False,
                                                                                   verbose = 1, 
                                                                                   monitor = "val_loss")
            print("---- Checkpoint Callback {} ----".format(checkpoint_callback))
            
        if not tokenizer or not model or not config:
            print("Not properly initialized....")
            
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, epsilon=epsilon)
        print("use_amp {}".format(use_amp))
        
        if use_amp:
            
            optimizer = tf.keras.mixed_precision.experimental.LossScaleOptimizer(optimizer, "dynamic")
            
        print("enable sagemaker debugger {}".format(enable_sagemaker_debugger))
        
        if enable_sagemaker_debugger:
            print("---- debugging ----")
            
            
            debugger_callback = smd.KerasHook.create_from_json_file()
            print("Debugger Callback {}".format(debugger_callback))
            
            callbacks.append(debugger_callback)
            
            optimizer = debugger_callback.wrap_optimizer(optimizer)
        
        if enable_tensorboard:
            tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=tensorboard_logs_path)
            print(" Tensorboard Callback {}".format(tensorboard_callback))
            callbacks.append(tensorboard_callback)
            
        print("Optimizer: {}".format(optimizer))
        
        loss = tf.keras.losses.MeanSquaredError()
        
        model.compile(optimizer = optimizer, loss = loss, metrics = [tf.keras.metrics.RootMeanSquaredError()])
        print("Compiled Model {}".format(model))
        
        print(model.summary())
        
        if run_validation:
            validation_data_filenames = gl(os.path.join(validation_data, "*.tfrecord"))
            
            print("validation data filenames {}".format(validation_data_filenames))
            
            validation_dataset = file_dataset_builder(channel = "validation",
                                                      input_filenames = validation_data_filenames,
                                                      pipe_mode = pipe_mode,
                                                      is_training = False,
                                                      drop_remainder = False,
                                                      batch_size = validation_batch_size,
                                                      epochs = epochs,
                                                      steps_per_epoch = validation_steps,
                                                      max_seq_length = max_seq_length).map(select_data_and_view_count)
            
            print("Starting Training and Validation")
            validation_dataset = validation_dataset.take(validation_steps)
            train_and_validation_hist = model.fit(train_dataset,
                                                  shuffle = True,
                                                  epochs = epochs,
                                                  initial_epoch = initial_epoch,
                                                  steps_per_epoch = train_steps_per_epoch,
                                                  validation_data = validation_dataset,
                                                  validation_steps = validation_steps,
                                                  callbacks = callbacks)
            print(train_and_validation_hist)
            
            
        else:
            print("Training without validation data")
            train_hist = model.fit(train_dataset,
                                   shuffle = True,
                                   epochs = epochs,
                                   initial_epoch = initial_epoch,
                                   steps_per_epoch = train_steps_per_epoch,
                                   callbacks = callbacks)
            print(train_hist)
            
        if run_test:
            test_data_files = gl(os.path.join(test-data, "*.tfrecord"))
            print("Test data files: {}".format(test_data_files))
            test_dataset = file_dataset_builder(channel = "test",
                                                      input_filenames = test_data_files,
                                                      pipe_mode = pipe_mode,
                                                      is_training = False,
                                                      drop_remainder = False,
                                                      batch_size = test_batch_size,
                                                      epochs = epochs,
                                                      steps_per_epoch = test_steps,
                                                      max_seq_length = max_seq_length).map(select_data_and_view_count)
            
            print("Starting Testing")
            
            test_hist = model.evaluate(test_dataset, steps = test_steps, callbacks = callbacks)
            
            print("Test History {}".format(test_hist))
            
        print("Tuned Model Path: {}".format(transformer_tuned_model_path))
        
        transformer_model.save_pretrained(transformer_tuned_model_path)
        
        print("Model inputs after saving: {}".format(model.inputs))
        
        print("Saved Tensorflow Model Path: {}".format(tensorflow_saved_model_path))
        
        model.save(tensorflow_saved_model_path, include_optimizer=False, overwrite=True, save_format = "tf")
        
        
        inference_path = os.path.join(model_directory,  "code/")
        print("Copying inference source files to path: {}".format(inference_path))
        print(glob(inference_path))
        
        os.system("cp -R ./test_data/ {}".format(model_directory))
    #if run_sample_predictions:
        
        #def predict(text):
            
            #encode_tokens = tokenizer.encode_plus(text, pad_to_max_length = True, max_length = max_seq_length,
                                                  #truncation = True, return_tensors = "tf")
            
            #input_ids = encode_tokens["input_ids"]
            
            #input_mask = encode_tokens["attention_mask"]
            
            
       # metrics_path = os.path.join(local_model_dir, "metrics/")
       # os.makedirs(metrics_path, exist_ok=True)
      # plt.savefig("{}/confusion_matrix.png".format(metrics_path))

       # report_dict = {
         #   "metrics": {
                #"rmse": {
                   # "value": rmse,
           #     },
        #    },
      #  }

       # evaluation_path = "{}/evaluation.json".format(metrics_path)
       # with open(evaluation_path, "w") as f:
         #   f.write(json.dumps(report_dict))
        
        
        
    
        
                                    
                
                
        
        
    
    
    
    
        


