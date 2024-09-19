from transformers import ViTImageProcessor, ViTForImageClassification, get_linear_schedule_with_warmup
from torchvision.io import read_image
from torchvision.transforms import v2
from torch.optim import AdamW
from tqdm import tqdm
from sklearn.metrics import f1_score
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import pandas as pd
import copy


class CancerDataset(Dataset):
    
    def __init__(self, path):
        dataset = pd.read_csv(path)
        self.X = dataset['Path'].tolist()
        self.y = torch.tensor(dataset.drop(columns='Path').values)
    
    def __getitem__(self, idx):
        path = self.X[idx]
        image = read_image(path)
        if image.shape[1] != 224 and image.shape[2] != 224:
            image = v2.functional.resize(image, (224, 224))

        return image, self.y[idx]
    
    def __len__(self):
        return len(self.X)
    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
# num_unfrozen_layers = [1,2,3]
# learning_rate = [5e-5, 1e-4, 5e-4]
# weight_decay = [0.01, 0.05]
# dropout_prob = [0.0, 0.1]
num_unfrozen_layers = 1
learning_rate = 1e-4
weight_decay = 0.01
dropout_prob = 0.1

batch_size = 32
epoch_num = 100

train_dataset = CancerDataset("preprocessed_data/train.csv")
test_dataset = CancerDataset("preprocessed_data/test.csv")
valid_dataset = CancerDataset("preprocessed_data/valid.csv")

torch.manual_seed(2543673)
train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size, shuffle=False)
valid_loader = DataLoader(valid_dataset, batch_size, shuffle=False)

checkpoint = "google/vit-base-patch16-224-in21k"
processor = ViTImageProcessor.from_pretrained(checkpoint)
model = ViTForImageClassification.from_pretrained(checkpoint, num_labels=8, problem_type='single_label_classification', hidden_dropout_prob=dropout_prob, attention_probs_dropout_prob= dropout_prob)
model.to(device)

for param in model.parameters():
    param.requires_grad = False
    
total_layer = len(model.vit.encoder.layer)

for layer in model.vit.encoder.layer[total_layer - num_unfrozen_layers:]:
    for param in layer.parameters():
        param.requires_grad = True
        
for param in model.classifier.parameters():
    param.requires_grad = True

optimiser = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr = learning_rate, weight_decay = weight_decay)
total_training_steps = epoch_num * len(train_loader)
scheduler = get_linear_schedule_with_warmup(optimiser, num_warmup_steps=0, num_training_steps = total_training_steps)

train_stats = {'train_accuracy' : [], 'train_loss' : [], 'train_f1' : [], 'valid_loss' : [], 'valid_accuracy' : [], 'valid_f1' : []}
min_val_loss = np.inf
no_epoch_update = 0
best_epoch = None
best_model = None
patience = 5


for idx in range(epoch_num):
    epoch_stats = {'train_accuracy' : [], 'train_loss' : [], 'train_f1' : [],'valid_f1' : [], 'valid_loss' : [], 'valid_accuracy' : []}
    model.train()
     
    with tqdm(train_loader) as pbar_train:
        for X_train, y_train in train_loader:
            X_train = processor(X_train, return_tensors='pt').to(device)
            y_train = y_train.argmax(1)
            
            outputs = model(**X_train, labels=y_train)
            train_loss = outputs.loss.to('cpu')
            train_predictions = outputs.logits.to('cpu').argmax(1)
            
            optimiser.zero_grad()
            train_loss.backward()
            optimiser.step()
            scheduler.step()
            
            train_accuracy = (train_predictions == y_train).sum() / len(y_train)
            train_f1 = f1_score(y_train, train_predictions, labels=[i for i in range(8)], average='macro', zero_division=0.0)
            epoch_stats['train_accuracy'].append(train_accuracy)
            epoch_stats['train_loss'].append(train_loss.detach().item())
            epoch_stats['train_f1'].append(train_f1)
            
            pbar_train.update(1)
    
    model.eval()
    
    with torch.no_grad(), tqdm(valid_loader) as pbar_valid:
        for X_valid, y_valid in valid_loader:
            X_valid = processor(X_valid, return_tensors='pt').to(device)
            y_valid = y_valid.argmax(1)
            
            outputs = model(**X_valid, labels=y_valid)
            valid_loss = outputs.loss.to('cpu')
            valid_predictions = outputs.logits.to('cpu').argmax(1)

            valid_accuracy = (valid_predictions== y_valid).sum() / len(y_valid)
            valid_f1 = f1_score(y_valid, valid_predictions, labels=[i for i in range(8)], average='macro', zero_division=0.0)
            epoch_stats['valid_accuracy'].append(valid_accuracy)
            epoch_stats['valid_loss'].append(valid_loss.item())
            epoch_stats['valid_f1'].append(valid_f1)
            
            pbar_valid.update(1)
            
    for key, value in epoch_stats.items():
        mean_val = np.mean(value)
        train_stats[key].append(mean_val)
    
    epoch_train_loss = train_stats['train_loss'][-1]
    epoch_train_accuracy = train_stats['train_accuracy'][-1]
    epoch_train_f1 = train_stats['train_f1'][-1]
    epoch_valid_loss = train_stats['valid_loss'][-1]
    epoch_valid_accuracy = train_stats['valid_accuracy'][-1]
    epoch_valid_f1 = train_stats['valid_f1'][-1]
    
    print("Epoch {:d} - train_loss: {:.4f}, train_accuracy: {:.4f}, train_f1: {:.4f}, valid_loss: {:.4f}, valid_accuracy: {:.4f}, valid_f1 : {:.4f}".format(idx,epoch_train_loss, epoch_train_accuracy, epoch_train_f1, epoch_valid_loss, epoch_valid_accuracy, epoch_valid_f1))
        
    if epoch_valid_loss < min_val_loss:
        min_val_loss = epoch_valid_loss
        best_epoch = idx
        no_epoch_update = 0
        best_model = copy.deepcopy(model.state_dict())
    else:
        no_epoch_update += 1
        
    if no_epoch_update >= patience:
        break
    
model.load_state_dict(best_model)
model.eval()

test_stats = {"test_accuracy" : [], "test_loss" : [], "test_f1" : []}
with torch.no_grad(), tqdm(len(test_loader)) as pbar_test:
    for X_test, y_test in test_loader:
        X_test = processor(X_test, return_tensors='pt').to(device)
        y_test = y_test.argmax(1)
        
        outputs = model(**X_test, labels=y_test)
        test_loss = outputs.loss.to('cpu')
        test_predictions = outputs.logits.to('cpu').argmax(1)
        
        test_accuracy = (test_predictions == y_test).sum() / len(y_test)
        test_f1 = f1_score(y_test,test_predictions, labels=[i for i in range(8)], average='macro', zero_division=0.0)
        test_stats['test_accuracy'].append(test_accuracy)
        test_stats['test_loss'].append(test_loss.item())
        test_stats['test_f1'].append(test_f1)
        
        pbar_test.update(1)
        
for key, value in test_stats.items():
    mean_val = np.mean(value)
    test_stats[key] = mean_val

model_info = {"best_param" : best_model, "best_epoch" : best_epoch, "train_stats": train_stats, "test_stats" : test_stats}    
torch.save(model_info, f"model/lr_{learning_rate}_decay_{weight_decay}_dropout_{dropout_prob}_unfrozen_{num_unfrozen_layers}.pt")
    
    
    