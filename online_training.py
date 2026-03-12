import os
import json
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from sklearn.model_selection import train_test_split
from datetime import datetime
from utils import *
from visualization import plot_training_summary

import argparse
import warnings

warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument("--dataset", type=str, default='nsl')
parser.add_argument("--epochs", type=int, default=4)
parser.add_argument("--epoch_1", type=int, default=1)
parser.add_argument("--percent", type=float, default=0.8)
parser.add_argument("--flip_percent", type=float, default=0.2)
parser.add_argument("--sample_interval", type=int, default=2000)
parser.add_argument("--cuda", type=str, default="0")

args = parser.parse_args()
dataset = args.dataset
epochs = args.epochs
epoch_1 = args.epoch_1
percent = args.percent
flip_percent = args.flip_percent
sample_interval = args.sample_interval
cuda_num = args.cuda

tem = 0.02
bs = 128
seed = 5009
seed_round = 1

if dataset == 'nsl':
    input_dim = 121
elif dataset == 'unsw':
    input_dim = 196
elif dataset == 'cic':
    input_dim = None
else:
    raise ValueError(f"Unsupported dataset: {dataset}")

if dataset == 'nsl':
    KDDTrain_dataset_path   = "NSL_pre_data/PKDDTrain+.csv"
    KDDTest_dataset_path    = "NSL_pre_data/PKDDTest+.csv"

    KDDTrain   =  load_data(KDDTrain_dataset_path)
    KDDTest    =  load_data(KDDTest_dataset_path)

    # 'labels2' means normal and abnormal, 'labels9' means 'attack_seen', 'attack_unseen', and normal
    # Create an instance of SplitData for 'nsl'
    splitter_nsl = SplitData(dataset='nsl')
    # Transform the data
    x_train, y_train = splitter_nsl.transform(KDDTrain, labels='labels2')
    x_test, y_test = splitter_nsl.transform(KDDTest, labels='labels2')

elif dataset == 'unsw':
    UNSWTrain_dataset_path   = "UNSW_pre_data/UNSWTrain.csv"
    UNSWTest_dataset_path    = "UNSW_pre_data/UNSWTest.csv"

    UNSWTrain   =  load_data(UNSWTrain_dataset_path)
    UNSWTest    =  load_data(UNSWTest_dataset_path)

    # Create an instance of SplitData for 'unsw'
    splitter_unsw = SplitData(dataset='unsw')

    # Transform the data
    x_train, y_train = splitter_unsw.transform(UNSWTrain, labels='label')
    x_test, y_test = splitter_unsw.transform(UNSWTest, labels='label')

else:  # cic
    CICTrain_dataset_path = "/kaggle/input/my-dataset-name/CIC_pre_data/CICTrain.csv"
    CICTest_dataset_path  = "/kaggle/input/my-dataset-name/CIC_pre_data/CICTest.csv"

    CICTrain = load_data(CICTrain_dataset_path)
    CICTest  = load_data(CICTest_dataset_path)

    splitter_cic = SplitData(dataset='cic')
    x_train, y_train = splitter_cic.transform(CICTrain, labels='label')
    x_test, y_test   = splitter_cic.transform(CICTest,  labels='label')
    input_dim = x_train.shape[1]
    print(f'CIC-IDS-2017 input_dim = {input_dim}')

# Convert to torch tensors
x_train, y_train = torch.FloatTensor(x_train), torch.LongTensor(y_train)
x_test, y_test = torch.FloatTensor(x_test), torch.LongTensor(y_test)

device = torch.device("cuda:"+cuda_num if torch.cuda.is_available() else "cpu")

criterion = CRCLoss(device, tem)

for i in range(seed_round):
    setup_seed(seed+i)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join('result', f'{dataset}_seed{seed+i}_{timestamp}')
    os.makedirs(run_dir, exist_ok=True)

    first_round_losses = []
    online_losses = []
    online_metrics = {}

    online_x_train, online_x_test, online_y_train, online_y_test = train_test_split(x_train, y_train, test_size=percent, random_state=seed+i)
    train_ds = TensorDataset(online_x_train, online_y_train)
    train_loader = torch.utils.data.DataLoader(
        dataset=train_ds, batch_size=bs, shuffle=True)
    
    num_of_first_train = online_x_train.shape[0]

    model = AE(input_dim).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr= 0.001)

####################### Stage 1: Offline Training #######################
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        num_batches = 0
        for j, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs = inputs.to(device)

            labels = labels.to(device)
            optimizer.zero_grad()

            features, recon_vec = model(inputs)
            loss = criterion(features,labels) + criterion(recon_vec,labels)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / max(num_batches, 1)
        first_round_losses.append(avg_loss)
        print(f'[Stage1] seed={seed+i}, epoch={epoch+1}/{epochs}, loss={avg_loss:.6f}')

    x_train = x_train.to(device)
    x_test = x_test.to(device)
    online_x_train, online_y_train  = online_x_train.to(device), online_y_train.to(device)

    x_train_this_epoch, x_test_left_epoch, y_train_this_epoch, y_test_left_epoch = online_x_train.clone(), online_x_test.clone().to(device), online_y_train.clone(), online_y_test.clone()

####################### Stage 2: Online Training #######################
    count = 0
    total_online_samples = len(x_test_left_epoch)
    total_online_steps = (total_online_samples + sample_interval - 1) // sample_interval
    y_train_detection = y_train_this_epoch
    y_test_left_labels = y_test_left_epoch.clone()

    while len(x_test_left_epoch) > 0:
        count += 1
        processed = min(count * sample_interval, total_online_samples)
        
        if len(x_test_left_epoch) < sample_interval:
            x_test_this_epoch = x_test_left_epoch.clone()
            y_true_this_step = y_test_left_labels.clone()
            x_test_left_epoch.resize_(0)
            y_test_left_labels.resize_(0)
        else:
            x_test_this_epoch = x_test_left_epoch[:sample_interval].clone()
            y_true_this_step = y_test_left_labels[:sample_interval].clone()
            x_test_left_epoch = x_test_left_epoch[sample_interval:]
            y_test_left_labels = y_test_left_labels[sample_interval:]

        with torch.no_grad():
            normal_data = online_x_train[(online_y_train == 0).squeeze()]
            enc, dec = model(normal_data)
            normal_temp = torch.mean(F.normalize(enc, p=2, dim=1), dim=0)
            normal_recon_temp = torch.mean(F.normalize(dec, p=2, dim=1), dim=0)
        predict_label = evaluate(normal_temp, normal_recon_temp, x_train_this_epoch, y_train_detection, x_test_this_epoch, 0, model)

        y_true_np = y_true_this_step.cpu().numpy()
        y_pred_np = predict_label.cpu().numpy() if isinstance(predict_label, torch.Tensor) else np.array(predict_label)
        batch_acc  = accuracy_score(y_true_np, y_pred_np)
        batch_prec = precision_score(y_true_np, y_pred_np, zero_division=0)
        batch_rec  = recall_score(y_true_np, y_pred_np, zero_division=0)
        batch_f1   = f1_score(y_true_np, y_pred_np, zero_division=0)
        online_metrics[count] = (batch_acc, batch_prec, batch_rec, batch_f1)

        print(f'[Stage2] seed={seed+i}, step={count}/{total_online_steps} '
              f'({100*processed/total_online_samples:.1f}%) | '
              f'Acc={batch_acc:.4f}  Prec={batch_prec:.4f}  '
              f'Rec={batch_rec:.4f}  F1={batch_f1:.4f}')

        y_test_pred_this_epoch = predict_label
        y_train_detection = torch.cat((y_train_detection.to(device), torch.tensor(y_test_pred_this_epoch).to(device)))
        num_zero = int(flip_percent * y_test_pred_this_epoch.shape[0])
        zero_indices = np.random.choice(y_test_pred_this_epoch.shape[0], num_zero, replace=False)
        y_test_pred_this_epoch[zero_indices] = 1 - y_test_pred_this_epoch[zero_indices]

        x_train_this_epoch = torch.cat((x_train_this_epoch.to(device), x_test_this_epoch.to(device)))
        y_train_this_epoch_temp = y_train_this_epoch.clone()
        y_train_this_epoch = torch.cat((y_train_this_epoch_temp.to(device), torch.tensor(y_test_pred_this_epoch).to(device)))

        train_ds = TensorDataset(x_train_this_epoch, y_train_this_epoch)
        
        train_loader = torch.utils.data.DataLoader(
            dataset=train_ds, batch_size=bs, shuffle=True)
        model.train()
        step_loss = 0.0
        step_batches = 0
        for epoch in range(epoch_1):
            for j, data in enumerate(train_loader, 0):
                inputs, labels = data
                inputs = inputs.to(device)

                labels = labels.to(device)
                optimizer.zero_grad()

                features, recon_vec = model(inputs)

                loss = criterion(features,labels) + criterion(recon_vec,labels)

                loss.backward()
                optimizer.step()

                step_loss += loss.item()
                step_batches += 1

        online_losses.append(step_loss / max(step_batches, 1))

################### Final Evaluation ###################
    with torch.no_grad():
        normal_data = online_x_train[(online_y_train == 0).squeeze()]
        enc, dec = model(normal_data)
        normal_temp = torch.mean(F.normalize(enc, p=2, dim=1), dim=0)
        normal_recon_temp = torch.mean(F.normalize(dec, p=2, dim=1), dim=0)

    res_en, res_de, res_final, y_pred_final = evaluate(
        normal_temp, normal_recon_temp, x_train_this_epoch, y_train_detection,
        x_test, y_test, model, return_predictions=True)

    print(f'\n{"=" * 60}')
    print(f'  Final Results - {dataset.upper()} seed={seed+i}')
    print(f'{"=" * 60}')
    print(f'  {"":12s} {"Acc":>8s} {"Prec":>8s} {"Recall":>8s} {"F1":>8s}')
    print(f'  {"-" * 44}')
    print(f'  {"Encoder":12s} {res_en[0]:>8.4f} {res_en[1]:>8.4f} {res_en[2]:>8.4f} {res_en[3]:>8.4f}')
    print(f'  {"Decoder":12s} {res_de[0]:>8.4f} {res_de[1]:>8.4f} {res_de[2]:>8.4f} {res_de[3]:>8.4f}')
    print(f'  {"Combined":12s} {res_final[0]:>8.4f} {res_final[1]:>8.4f} {res_final[2]:>8.4f} {res_final[3]:>8.4f}')
    print(f'{"=" * 60}')

    # ==================== Save results to result/ ====================
    metric_names = ['accuracy', 'precision', 'recall', 'f1']
    run_result = {
        'config': {
            'dataset': dataset,
            'seed': seed + i,
            'epochs': epochs,
            'epoch_1': epoch_1,
            'percent': percent,
            'flip_percent': flip_percent,
            'sample_interval': sample_interval,
            'batch_size': bs,
            'temperature': tem,
            'input_dim': input_dim,
            'num_first_train': num_of_first_train,
            'timestamp': timestamp,
        },
        'stage1_losses': first_round_losses,
        'stage2_losses': online_losses,
        'online_metrics': {
            str(step): dict(zip(metric_names, [float(v) for v in vals]))
            for step, vals in online_metrics.items()
        },
        'final_results': {
            'encoder':  dict(zip(metric_names, [float(v) for v in res_en])),
            'decoder':  dict(zip(metric_names, [float(v) for v in res_de])),
            'combined': dict(zip(metric_names, [float(v) for v in res_final])),
        },
    }

    with open(os.path.join(run_dir, 'metrics.json'), 'w') as f:
        json.dump(run_result, f, indent=2)

    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, os.path.join(run_dir, 'model.pth'))

    y_test_np = y_test.cpu().numpy() if isinstance(y_test, torch.Tensor) else np.array(y_test)
    np.savez(
        os.path.join(run_dir, 'predictions.npz'),
        y_true=y_test_np,
        y_pred=y_pred_final,
    )

    plot_training_summary(
        first_round_losses=first_round_losses,
        online_losses=online_losses,
        online_metrics=online_metrics,
        final_encoder=res_en,
        final_decoder=res_de,
        final_combined=res_final,
        y_test_true=y_test_np,
        y_test_pred=y_pred_final,
        dataset=dataset,
        seed=seed+i,
        save_dir=run_dir,
    )

    print(f'  [Saved] All results -> {run_dir}/')
