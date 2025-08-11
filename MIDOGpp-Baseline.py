import comet_ml
from comet_ml import Experiment
from comet_ml import start, login
from comet_ml.integration.pytorch import watch

import os
import platform
import sys
from packaging import version
import pandas as pd

import sklearn
from sklearn.metrics import roc_auc_score

from tqdm import tqdm
import matplotlib.pyplot as plt
import copy
import argparse
from glob import glob
import random
import datetime

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as torch_data
import torchvision.transforms as torch_transforms
from torchvision.transforms import ToTensor
import torch.nn.functional as F
from torch.utils.data import Dataset
from albumentations.pytorch import ToTensorV2

import monai
import monai.data as monai_data
import monai.transforms as monai_transforms
import monai.networks.nets as monai_nn
from monai.networks.nets import ResNet, DenseNet, ViT

from skimage import io
import os
from PIL import Image
import albumentations as A
import numpy as np


experiment = Experiment(
    api_key="LifP49eHZV49oQCylB3x2Sy2n",
    project_name="challenge",
)

login()
experiment = start()

timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
print(timestamp)

job_id = os.getenv('SLURM_JOB_ID')
if job_id is None:
    job_id = os.getenv('PBS_JOBID') # Fallback for PBS/Torque
if job_id:
    print(f"CHPC Job ID: {job_id}")
    experiment.log_parameter("chpc_job_id", job_id)
else:
    print("Not running within a CHPC/SLURM/PBS job; Job ID not found.")

print(f"Script Timestamp: {timestamp}")

##### Check PyTorch has access to GPU #####
def check_gpu_access():
    # Check the current operating system
    os_name = platform.system()
    device = 'cpu'

    # Switch case based on the operating system
    if os_name == 'Windows':
        # Windows specific GPU check
        if torch.cuda.is_available():
            device = 'cuda'
    elif os_name == 'Linux':
        # Linux specific GPU check
        if torch.cuda.is_available():
            device = 'cuda'
    elif os_name == 'Darwin':
        # Mac specific GPU check
        if torch.backends.mps.is_available():
            device = 'mps'
        elif torch.cuda.is_available():
            device = 'cuda'

    return {
        "Operating System": os_name,
        "Device": device
    }

class MIDOG25Loader(Dataset):
    def __init__(self,dataframe, root, transform=None):
        self.root = root
        self.transforms = transform
        self.imgNames = dataframe
        
    def __getitem__(self, index):
        row = self.imgNames.iloc[index]
        
        atypical = str(row['majority']).strip().upper() == "AMF"
        image_name = row['image_id']

        image_path = os.path.join(self.root, 'MIDOG25_Binary_Classification_Train_Set', image_name)
        image = Image.open(image_path).convert("RGB")

        if self.transforms:
            image = self.transforms(image)

        label = int(atypical)
        return image, label

    
    def __len__(self):
        return len(self.imgNames)

    
def evaluation(args,model,data_loader, device):
    criteria = nn.CrossEntropyLoss()

    model.eval()
    eval_loss = 0
    num_correct = 0
    num_examples = 0
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for examples, label in (progress := tqdm(data_loader, desc="evaluation", file=sys.stdout)):
            examples = examples.to(device)
            label = label.to(device)

            logits = model(examples)
            if isinstance(logits, tuple):
                logits = logits[0]

            loss = criteria(logits, label)
            predictions = logits.max(1).indices
            num_correct += (predictions == label).sum().item()

            num_examples += label.shape[0]
            eval_loss += loss.item() * label.shape[0]

            prob = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
            all_probs.extend(prob.flatten())
            all_labels.extend(label.cpu().numpy().flatten())

            avg_loss = eval_loss / num_examples
            acc = num_correct / num_examples

            progress.set_postfix_str(f"average loss: {avg_loss:.5f}, accuracy: {acc:.4f} ")

    auc = 0.0
    if len(all_labels) > 0 and len(np.unique(all_labels)) > 1: 
        auc = roc_auc_score(all_labels, all_probs)
        print(f"AUC: {auc:.4f}") 

        
    return avg_loss, acc, auc
    
def train(args,model,train_loader,val_loader, device):

    optimizer = torch.optim.Adam(model.parameters(), lr= args.learningRate)
    criteria = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    metrics = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "val_auc": []}

    for num_epoch in range(0, args.epochs):
        model.train()
        epoch_loss = 0
        correct = 0
        num_examples = 0

        for example, label in (progress_bar := tqdm(train_loader, desc=f"training epoch: {num_epoch}", file=sys.stdout)):
            optimizer.zero_grad()
            example = example.to(device)
            label = label.to(device)

            logits = model(example)
            if isinstance(logits, tuple):
                logits = logits[0]

            loss = criteria(logits, label)
            loss.backward()
            optimizer.step()
            correct += (logits.max(1).indices == label).sum().item()

            num_examples += label.shape[0]
            epoch_loss += loss.item() * label.shape[0]

            train_loss = epoch_loss/num_examples
            train_acc = correct/num_examples

            progress_bar.set_postfix_str(f"Epoch {num_epoch}: train_loss={train_loss:.5f}, train_acc={train_acc:.4f}")
        
        print("-------------------------")
        val_loss, val_acc, _ = evaluation(args, model, val_loader, device)
        print(f"Epoch= {num_epoch} val_loss= {val_loss:.5f}, val_acc= {val_acc:.4f}")
        print("-------------------------")

        metrics["train_loss"].append(train_loss)
        metrics["train_acc"].append(train_acc)
        metrics["val_loss"].append(val_loss)
        metrics["val_acc"].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model = copy.deepcopy(model.state_dict())

    torch.save(best_model, f"best_model{timestamp}.pth")
        

def main(args):
    # Execute the function and print the results
    gpu_access = check_gpu_access()
    device = gpu_access['Device']
    device = torch.device(device)
    print(f"Using device: {device}")
    experiment.log_parameters(args)
    print(args)

    random.seed(42)

    data_transform = torch_transforms.Compose([
        torch_transforms.ToTensor(),
        torch_transforms.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0])
    ])

    data_path = os.path.join(os.getcwd(), "MIDOG2025-Binary-Classification")
    path = os.path.join(os.getcwd(), "MIDOG2025-Binary-Classification", "MIDOG25_Atypical_Classification_Train_Set.csv")
    df = pd.read_csv(path,header=0)

    indices = list(range(len(df)))
    random.shuffle(indices)

    train_size = int(0.8 * len(df))
    val_size = int(0.1 * len(df))

    train_indices = indices[:train_size]
    val_indices   = indices[train_size:train_size + val_size]
    test_indices  = indices[train_size + val_size:]

    train_df = df.iloc[train_indices]
    val_df   = df.iloc[val_indices]
    test_df  = df.iloc[test_indices]

    train_data = MIDOG25Loader(train_df, data_path, data_transform)
    val_data = MIDOG25Loader(val_df, data_path, data_transform)
    test_data = MIDOG25Loader(test_df, data_path, data_transform)

    train_loader = torch_data.DataLoader(dataset= train_data, batch_size= args.batchSize, shuffle=True)
    val_loader = torch_data.DataLoader(dataset= val_data, batch_size= args.batchSize, shuffle=True)
    test_loader = torch_data.DataLoader(dataset= test_data, batch_size= args.batchSize, shuffle=True)

    model = ResNet(
            block= 'basic',
            layers= [2, 2, 2, 2],
            block_inplanes=[64, 128, 256, 512],
            spatial_dims= 2, 
            n_input_channels= 3,
            num_classes= 2)
    
    model.to(device)
    watch(model)

    print("Dataset: MIDOGpp")
    print("Model: ResNet")

    ##### Training Process #####
    train(args, model, train_loader, val_loader, device)

    ##### Testing Process #####
    # Reload the best saved model from training and report testing accuracy.
    model.load_state_dict(torch.load(f"best_model{timestamp}.pth"))

    evaluation(args, model, test_loader, device)

script_dir = os.getcwd()

output_filename = f"ResNet-{timestamp}.txt"
output_filepath = os.path.join(script_dir, output_filename)

original_stdout = sys.stdout
original_stderr = sys.stderr

with open(output_filepath, 'w') as f:
    sys.stdout = f

    try:
        batch = 4
        lr = 1e-3
        epoch = 100
        print(f"batch size: {batch}, lr: {lr}, epoch: {epoch}")
        main(argparse.Namespace(batchSize= batch, learningRate=lr, epochs=epoch))

    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr

print(f"Summary results written to {output_filepath}")
