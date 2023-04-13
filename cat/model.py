import time
import random
import pandas as pd
from glob import glob
import pprint
import argparse
import json
import subprocess
import sys
import os
import csv
subprocess.check_call([sys.executable, "-m", "pip", "install", "transformers==3.5.1"])

subprocess.check_call([sys.executable, "-m", "pip", "install", "scikit-learn==0.23.1"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "matplotlib==3.2.1"])

import tensorflow as tf
import pandas as pd
import numpy as np
import s3fs
import io

import itertools
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report



from transformers import DistilBertTokenizer
from transformers import DistilBertConfig
from transformers import TFDistilBertForSequenceClassification

from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model


CLASSES = [0,1]


def select_data_and_label_from_record(record):
    x = {"input_ids": record["title_input_ids"], "input_mask": record["title_input_mask"], "segment_ids": record["segment_ids"]}

    y = record["cat_view_count_id"]

    return (x, y)


def file_based_input_dataset_builder(channel,input_filenames,pipe_mode, is_training, drop_remainder, batch_size, epochs,steps_per_epoch,max_seq_length):



    if pipe_mode:
        print("***** Using pipe_mode with channel {}".format(channel))
        from sagemaker_tensorflow import PipeModeDataset

        dataset = PipeModeDataset(channel=channel, record_format="TFRecord")
    else:
        print("***** Using input_filenames {}".format(input_filenames))
        dataset = tf.data.TFRecordDataset(input_filenames)

    dataset = dataset.repeat(epochs * steps_per_epoch * 100)

    name_to_features = {
        "title_input_ids": tf.io.FixedLenFeature([max_seq_length], tf.int64),
        "title_input_mask": tf.io.FixedLenFeature([max_seq_length], tf.int64),
        "segment_ids": tf.io.FixedLenFeature([max_seq_length], tf.int64),
        "cat_view_count_id": tf.io.FixedLenFeature([], tf.int64),
    }

    def _decode_record(record, name_to_features):
        """Decodes a record to a TensorFlow example."""
        record = tf.io.parse_single_example(record, name_to_features)
        return record

    dataset = dataset.apply(
        tf.data.experimental.map_and_batch(
            lambda record: _decode_record(record, name_to_features),
            batch_size=batch_size,
            drop_remainder=drop_remainder,
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
        )
    )



    dataset = dataset.shuffle(buffer_size=1000, reshuffle_each_iteration=True)

    row_count = 0
    print("**************** {} *****************".format(channel))
    for row in dataset.as_numpy_iterator():
        print(row)
        if row_count == 5:
            break
        row_count = row_count + 1

    return dataset


def load_checkpoint_model(checkpoint_path):
    import glob
    import os

    glob_pattern = os.path.join(checkpoint_path, "*.h5")
    print("glob pattern {}".format(glob_pattern))

    list_of_checkpoint_files = glob.glob(glob_pattern)
    print("List of checkpoint files {}".format(list_of_checkpoint_files))

    latest_checkpoint_file = max(list_of_checkpoint_files)
    print("Latest checkpoint file {}".format(latest_checkpoint_file))

    initial_epoch_number_str = latest_checkpoint_file.rsplit("_", 1)[-1].split(".h5")[0]
    initial_epoch_number = int(initial_epoch_number_str)

    loaded_model = TFDistilBertForSequenceClassification.from_pretrained(latest_checkpoint_file, config=config)

    print("loaded_model {}".format(loaded_model))
    print("initial_epoch_number {}".format(initial_epoch_number))

    return loaded_model, initial_epoch_number


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_data", type=str, default=os.environ["SM_CHANNEL_TRAIN"])
    parser.add_argument("--validation_data", type=str, default=os.environ["SM_CHANNEL_VALIDATION"])
    parser.add_argument("--output_dir", type=str, default=os.environ["SM_OUTPUT_DIR"])
    parser.add_argument("--hosts", type=list, default=json.loads(os.environ["SM_HOSTS"]))
    parser.add_argument("--current_host", type=str, default=os.environ["SM_CURRENT_HOST"])
    parser.add_argument("--num_gpus", type=int, default=os.environ["SM_NUM_GPUS"])
    parser.add_argument("--checkpoint_base_path", type=str, default="/opt/ml/checkpoints")
    parser.add_argument("--use_xla", type=eval, default=False)
    parser.add_argument("--use_amp", type=eval, default=False)
    parser.add_argument("--max_seq_length", type=int, default=80)
    parser.add_argument("--train_batch_size", type=int, default=128)
    parser.add_argument("--validation_batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=0.00003)
    parser.add_argument("--epsilon", type=float, default=0.00000001)
    parser.add_argument("--train_steps_per_epoch", type=int, default=None)
    parser.add_argument("--validation_steps", type=int, default=None)
    parser.add_argument("--freeze_bert_layer", type=eval, default=False)
    parser.add_argument("--enable_sagemaker_debugger", type=eval, default=False)
    parser.add_argument("--run_validation", type=eval, default=False)
    parser.add_argument("--run_sample_predictions", type=eval, default=False)
    parser.add_argument("--enable_tensorboard", type=eval, default=False)
    parser.add_argument("--enable_checkpointing", type=eval, default=False)
    parser.add_argument("--local_bucket", type=str, default = None)
    parser.add_argument("--output_data_dir", type=str, default=os.environ["SM_OUTPUT_DATA_DIR"]) 
    



    args, _ = parser.parse_known_args()
    print("Args:")
    print(args)

    env_var = os.environ
    print("Environment Variables:")
    pprint.pprint(dict(env_var), width=1)

    print("SM_TRAINING_ENV {}".format(env_var["SM_TRAINING_ENV"]))
    sm_training_env_json = json.loads(env_var["SM_TRAINING_ENV"])
    is_master = sm_training_env_json["is_master"]
    print("is_master {}".format(is_master))

    train_data = args.train_data
    print("train_data {}".format(train_data))
    validation_data = args.validation_data
    print("validation_data {}".format(validation_data))

    local_model_dir = os.environ["SM_MODEL_DIR"]
    output_dir = args.output_dir
    print("output_dir {}".format(output_dir))
    hosts = args.hosts
    print("hosts {}".format(hosts))
    current_host = args.current_host
    print("current_host {}".format(current_host))
    num_gpus = args.num_gpus
    print("num_gpus {}".format(num_gpus))
    job_name = os.environ["SAGEMAKER_JOB_NAME"]
    print("job_name {}".format(job_name))
    use_xla = args.use_xla
    print("use_xla {}".format(use_xla))
    use_amp = args.use_amp
    print("use_amp {}".format(use_amp))
    max_seq_length = args.max_seq_length
    print("max_seq_length {}".format(max_seq_length))
    train_batch_size = args.train_batch_size
    print("train_batch_size {}".format(train_batch_size))
    validation_batch_size = args.validation_batch_size
    print("validation_batch_size {}".format(validation_batch_size))

    epochs = args.epochs
    print("epochs {}".format(epochs))
    learning_rate = args.learning_rate
    print("learning_rate {}".format(learning_rate))
    epsilon = args.epsilon
    print("epsilon {}".format(epsilon))
    train_steps_per_epoch = args.train_steps_per_epoch
    print("train_steps_per_epoch {}".format(train_steps_per_epoch))
    validation_steps = args.validation_steps
    print("validation_steps {}".format(validation_steps))

    freeze_bert_layer = args.freeze_bert_layer
    print("freeze_bert_layer {}".format(freeze_bert_layer))
    enable_sagemaker_debugger = args.enable_sagemaker_debugger
    print("enable_sagemaker_debugger {}".format(enable_sagemaker_debugger))
    run_validation = args.run_validation
    print("run_validation {}".format(run_validation))

    run_sample_predictions = args.run_sample_predictions
    print("run_sample_predictions {}".format(run_sample_predictions))
    enable_tensorboard = args.enable_tensorboard
    print("enable_tensorboard {}".format(enable_tensorboard))
    enable_checkpointing = args.enable_checkpointing
    print("enable_checkpointing {}".format(enable_checkpointing))
    local_bucket = args.local_bucket
    print("Local bucket {}".format(local_bucket))

    checkpoint_base_path = args.checkpoint_base_path
    print("checkpoint_base_path {}".format(checkpoint_base_path))

    if is_master:
        checkpoint_path = checkpoint_base_path
    else:
        checkpoint_path = "/tmp/checkpoints"
    print("checkpoint_path {}".format(checkpoint_path))

    pipe_mode_str = os.environ.get("SM_INPUT_DATA_CONFIG", "")
    pipe_mode = pipe_mode_str.find("Pipe") >= 0
    print("Using pipe_mode: {}".format(pipe_mode))

    transformer_fine_tuned_model_path = os.path.join(local_model_dir, "transformers/fine-tuned/")
    os.makedirs(transformer_fine_tuned_model_path, exist_ok=True)

    tensorflow_saved_model_path = os.path.join(local_model_dir, "tensorflow/saved_model/0")
    os.makedirs(tensorflow_saved_model_path, exist_ok=True)

    tensorboard_logs_path = os.path.join(local_model_dir, "tensorboard/")
    os.makedirs(tensorboard_logs_path, exist_ok=True)



    distributed_strategy = tf.distribute.MirroredStrategy()
    with distributed_strategy.scope():
        tf.config.optimizer.set_jit(use_xla)
        tf.config.optimizer.set_experimental_options({"auto_mixed_precision": use_amp})

        train_data_filenames = glob(os.path.join(train_data, "*.tfrecord"))
        print("train_data_filenames {}".format(train_data_filenames))
        train_dataset = file_based_input_dataset_builder(
            channel="train",
            input_filenames=train_data_filenames,
            pipe_mode=pipe_mode,
            is_training=True,
            drop_remainder=False,
            batch_size=train_batch_size,
            epochs=epochs,
            steps_per_epoch=train_steps_per_epoch,
            max_seq_length=max_seq_length,
        ).map(select_data_and_label_from_record)

        tokenizer = None
        config = None
        model = None
        transformer_model = None

        successful_download = False
        retries = 0
        while retries < 5 and not successful_download:
            try:
                tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
                config = DistilBertConfig.from_pretrained(
                    "distilbert-base-uncased",
                    num_labels=len(CLASSES),
                    id2label={0: 0, 1:1},
                    label2id={0:0, 1:1},
                )

                transformer_model = TFDistilBertForSequenceClassification.from_pretrained(
                    "distilbert-base-uncased", config=config
                )

                input_ids = tf.keras.layers.Input(shape=(max_seq_length,), name="title_input_ids", dtype="int32")
                input_mask = tf.keras.layers.Input(shape=(max_seq_length,), name="title_input_mask", dtype="int32")

                embedding_layer = transformer_model.distilbert(input_ids, attention_mask=input_mask)[0]
                X = tf.keras.layers.Bidirectional(
                    tf.keras.layers.LSTM(50, return_sequences=True, dropout=0.1, recurrent_dropout=0.1)
                )(embedding_layer)
                X = tf.keras.layers.GlobalMaxPool1D()(X)
                X = tf.keras.layers.Dense(50, activation="relu")(X)
                X = tf.keras.layers.Dropout(0.2)(X)
                X = tf.keras.layers.Dense(1, activation="sigmoid")(X)

                model = tf.keras.Model(inputs=[input_ids, input_mask], outputs=X)

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

        initial_epoch_number = 0

        if enable_checkpointing:
            print("***** Checkpoint enabled *****")

            os.makedirs(checkpoint_path, exist_ok=True)
            if os.listdir(checkpoint_path):
                print("***** Found checkpoint *****")
                print(checkpoint_path)
                model, initial_epoch_number = load_checkpoint_model(checkpoint_path)
                print("***** Using checkpoint model {} *****".format(model))

            checkpoint_callback = ModelCheckpoint(
                filepath=os.path.join(checkpoint_path, "tf_model_{epoch:05d}.h5"),
                save_weights_only=False,
                verbose=1,
                monitor="val_accuracy",
            )
            print("*** CHECKPOINT CALLBACK {} ***".format(checkpoint_callback))
            callbacks.append(checkpoint_callback)

        if not tokenizer or not model or not config:
            print("Not properly initialized...")

        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, epsilon=epsilon)
        print("** use_amp {}".format(use_amp))
        if use_amp:
            optimizer = tf.keras.mixed_precision.experimental.LossScaleOptimizer(optimizer, "dynamic")

        print("enable_sagemaker_debugger {}".format(enable_sagemaker_debugger))
        if enable_sagemaker_debugger:
            print("*** DEBUGGING ***")
            import smdebug.tensorflow as smd

            debugger_callback = smd.KerasHook.create_from_json_file()
            print("*** DEBUGGER CALLBACK {} ***".format(debugger_callback))
            callbacks.append(debugger_callback)
            optimizer = debugger_callback.wrap_optimizer(optimizer)

        if enable_tensorboard:
            tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=tensorboard_logs_path)
            print("*** TENSORBOARD CALLBACK {} ***".format(tensorboard_callback))
            callbacks.append(tensorboard_callback)

        print("*** OPTIMIZER {} ***".format(optimizer))

        loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        metric = tf.keras.metrics.BinaryAccuracy("accuracy")

        model.compile(optimizer=optimizer, loss=loss, metrics=[metric])
        print("Compiled model {}".format(model))
        print(model.summary())

        if run_validation:
            validation_data_filenames = glob(os.path.join(validation_data, "*.tfrecord"))
            print("validation_data_filenames {}".format(validation_data_filenames))
            validation_dataset = file_based_input_dataset_builder(
                channel="validation",
                input_filenames=validation_data_filenames,
                pipe_mode=pipe_mode,
                is_training=False,
                drop_remainder=False,
                batch_size=validation_batch_size,
                epochs=epochs,
                steps_per_epoch=validation_steps,
                max_seq_length=max_seq_length,
            ).map(select_data_and_label_from_record)

            print("Starting Training and Validation...")
            validation_dataset = validation_dataset.take(validation_steps)
            train_and_validation_history = model.fit(
                train_dataset,
                shuffle=True,
                epochs=epochs,
                initial_epoch=initial_epoch_number,
                steps_per_epoch=train_steps_per_epoch,
                validation_data=validation_dataset,
                validation_steps=validation_steps,
                callbacks=callbacks,
            )
            print(train_and_validation_history)
        else: 
            print("Starting Training (Without Validation)...")
            train_history = model.fit(
                train_dataset,
                shuffle=True,
                epochs=epochs,
                initial_epoch=initial_epoch_number,
                steps_per_epoch=train_steps_per_epoch,
                callbacks=callbacks,
            )
            print(train_history)

     

        print("transformer_fine_tuned_model_path {}".format(transformer_fine_tuned_model_path))
        transformer_model.save_pretrained(transformer_fine_tuned_model_path)
        print("Model inputs after save_pretrained: {}".format(model.inputs))

        print("tensorflow_saved_model_path {}".format(tensorflow_saved_model_path))
        model.save(tensorflow_saved_model_path, include_optimizer=False, overwrite=True, save_format="tf")


    if run_sample_predictions:

        def predict(title):
            encode_plus_tokens = tokenizer.encode_plus(
                title, pad_to_max_length=True, max_length=max_seq_length, truncation=True, return_tensors="tf")
            
            title_input_ids = encode_plus_tokens["input_ids"]

            
            title_input_mask = encode_plus_tokens["attention_mask"]

            outputs = model.predict(x=(title_input_ids, title_input_mask))

            prediction = [{"label": config.id2label[item.argmax()], "score": item.max().item()} for item in outputs]

            return prediction[0]["label"]
        
        def predict_proba(title):
            encode_plus_tokens = tokenizer.encode_plus(title, pad_to_max_length=True, max_length=max_seq_length, truncation=True, return_tensors="tf")
            title_input_ids = encode_plus_tokens["input_ids"]
            title_input_mask = encode_plus_tokens["attention_mask"]
            
            preds = model.predict(x=(title_input_ids, title_input_mask)).logits
            res = tf.nn.sigmoid(preds, axis=1).numpy()      
    
            return res

        print(
            """Valorant VCT Replay""",
            predict("""Valorant VCT Replay"""),
        )

        print("""FaZe House""", predict("""FaZe House"""))

        print(
            """Mario Maker Speed Runs""",
            predict("""Mario Maker Speed Runs"""),
        )
        forres_df = pd.read_csv("s3://{}/youtubeStatistics/cat_dfs/test/gaming_test.csv".format(local_bucket))[["title", "video_id", "view_count"]]
        df_test_reviews = pd.read_csv("s3://{}/youtubeStatistics/cat_dfs/test/gaming_test.csv".format(local_bucket))[["title", "cat_view_count"]]
        df_test_reviews.shape
        df_test_reviews.head()

        y_test = df_test_reviews["title"].map(predict)
        y_test

        y_actual = df_test_reviews["cat_view_count"]
        y_actual
        
        #y_prob = pd.DataFrame(df_test_reviews["title"].map(predict_proba), columns = "test_probability")
        
       # print(y_prob.head())
    
        results_df = pd.DataFrame({"Prediction": y_test, "Actual": y_actual, "title": forres_df["title"], "video_id": forres_df["video_id"], "view_count": forres_df["view_count"]})
        
        

        print(classification_report(y_true=y_test, y_pred=y_actual))

        accuracy = accuracy_score(y_true=y_test, y_pred=y_actual)
        print("Test accuracy: ", accuracy)


        def plot_conf_mat(cm, classes, title, cmap=plt.cm.Greens):
            print(cm)
            plt.imshow(cm, interpolation="nearest", cmap=cmap)
            plt.title(title)
            plt.colorbar()
            tick_marks = np.arange(len(classes))
            plt.xticks(tick_marks, classes, rotation=45)
            plt.yticks(tick_marks, classes)

            fmt = "d"
            thresh = cm.max() / 2.0
            for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
                plt.text(
                    j,
                    i,
                    format(cm[i, j], fmt),
                    horizontalalignment="center",
                    color="black" if cm[i, j] > thresh else "black",
                )

                plt.tight_layout()
                plt.ylabel("True label")
                plt.xlabel("Predicted label")

        cm = confusion_matrix(y_true=y_test, y_pred=y_actual)

        plt.figure()
        fig, ax = plt.subplots(figsize=(10, 5))
        plot_conf_mat(cm, classes=["0","1"], title="Confusion Matrix")

        plt.show()
        img_data = io.BytesIO()
        plt.savefig(img_data, format='png', bbox_inches='tight')
        img_data.seek(0)
        image_name = "confusion_matrix.png"
        
        s3 = s3fs.S3FileSystem(anon=False)  # Uses default credentials
        with s3.open('s3://{}/youtubeStatistics/'.format(local_bucket)+image_name, 'wb') as f:
            f.write(img_data.getbuffer())
        results_df.to_csv("s3://{}/youtubeStatistics/results_df.csv".format(local_bucket))


        metrics_path = os.path.join(local_model_dir, "metrics/")
        os.makedirs(metrics_path, exist_ok=True)

        report_dict = {
            "metrics": {
                "accuracy": {
                    "value": accuracy,
                },
            },
        }

        evaluation_path = "{}/evaluation.json".format(metrics_path)
        with open(evaluation_path, "w") as f:
            f.write(json.dumps(report_dict))