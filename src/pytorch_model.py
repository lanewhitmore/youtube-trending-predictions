
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
#subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'tensorflow==2.3.0'])
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'torch==1.9.0'])
subprocess.check_call([sys.executable, "-m", "pip", "install", "transformers==3.5.1"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "scikit-learn==0.23.1"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "matplotlib==3.2.1"])
#subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'sagemaker-tensorflow==2.1.0.1.0.0'])
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'smdebug==0.9.3'])

#import tensorflow as tf
#from tensorflow.keras.callbacks import CSVLogger
import smdebug.tensorflow as smd
#from tensorflow.keras.callbacks import ModelCheckpoint
#from tensorflow.keras.models import load_model
import pandas as pd
import numpy as np
import boto3

from transformers import DistilBertTokenizer
from transformers import DistilBertConfig
from transformers import DistilBertModel
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.nn.utils.clip_grad import clip_grad_norm


from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, mean_absolute_error, r2_score






def create_dataloaders(inputs, masks, labels, batch_size):
    input_tensor = torch.tensor(inputs)
    mask_tensor = torch.tensor(masks)
    labels_tensor = torch.tensor(labels)
    dataset = TensorDataset(input_tensor, mask_tensor, 
                                               labels_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, 
                                                shuffle=True)
    return dataloader







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
    
def transform_csv_to_tensor_to_dataloader(file):
    
    df = pd.read_csv(file)
    
    train, test = train_test_split(df, test_size = 0.1)
    
    
    batch_size =64
    
    train_dataloader = create_dataloaders(train.title_input_ids, train.title_input_masks, 
                                                                       train.view_count, batch_size)
    
    test_dataloader = create_dataloaders(test.title_input_ids, test.title_input_masks, 
                                                                      test.view_count, batch_size)
    
    return train, test


class distilBertRegressor(nn.Module):

    def __init__(self, drop_rate=0.2):

        super(distilBertRegressor, self).__init__()
        D_in, D_out = 768, 1

        self.distilbert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.regressor = nn.Sequential(
            nn.Dropout(drop_rate),
            nn.Linear(D_in, D_out))    
        def forward(self, input_ids, attention_masks):

            outputs = self.distilbert(input_ids, attention_masks)
            class_label_output = outputs[1]
            outputs = self.regressor(class_label_output)
            return outputs
        
def train(model, optimizer, scheduler, loss_function, epochs, train_dataloader, device, clip_value=2):
    for epoch in range(epochs):
        print(epoch)
        best_loss = 1e10
        model.train()
        for step, batch in enumerate(train_dataloader): 
            print(step)  
            batch_inputs, batch_masks, batch_labels = tuple(b.to(device) for b in batch)
            model.zero_grad()
            outputs = model(batch_inputs, batch_masks)           
            loss = loss_function(outputs.squeeze(), 
                             batch_labels.squeeze())
            loss.backward()
            clip_grad_norm(model.parameters(), clip_value)
            optimizer.step()
            scheduler.step()
                
    return model


def predict(model, dataloader, device):
    model.eval()
    output = []
    for batch in dataloader:
        batch_inputs, batch_masks, _ = tuple(b.to(device) for b in batch)
        with torch.no_grad():
            output += model(batch_inputs, batch_masks).view(1,-1).tolist()[0]
    return output


def evaluate(model, loss_function, test_dataloader, device):
    model.eval()
    test_loss, test_r2 = [], []
    for batch in test_dataloader:
        batch_inputs, batch_masks, batch_labels = \
                                 tuple(b.to(device) for b in batch)
        with torch.no_grad():
            outputs = model(batch_inputs, batch_masks)
        loss = loss_function(outputs, batch_labels)
        test_loss.append(loss.item())
        r2 = r2_score(outputs, batch_labels)
        test_r2.append(r2.item())
    return test_loss, test_r2


def r2_score(outputs, labels):
    labels_mean = torch.mean(labels)
    ss_tot = torch.sum((labels - labels_mean) ** 2)
    ss_res = torch.sum((labels - outputs) ** 2)
    r2 = 1 - ss_res / ss_tot
    return r2


    

if __name__ == "__main__":
    
    train_loader, test_loader = transform_csv_to_tensor_to_dataloader("s3://sagemaker-us-east-1-492991381452/youtubeStatistics/dfs/gaming.csv")


    optimizer = AdamW(model.parameters(),
                                       lr=5e-5,
                                       eps=1e-8)        
    
    
    epochs = 5
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    loss_function = nn.MSELoss()
    
    print("Training Model.....")
    model = distilBertRegressor(drop_rate=0.2)
    
    
    model = train(model, optimizer, scheduler, loss_function, epochs, 
                             train_loader, device, clip_value=2)