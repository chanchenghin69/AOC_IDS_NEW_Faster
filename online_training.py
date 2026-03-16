import os
import json
import time
import shutil
import argparse
import warnings
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from utils import *
from visualization import plot_training_summary

warnings.filterwarnings("ignore")

START_TIME = time.time()
MAX_RUN_SECONDS = 11.5 * 3600   # Kaggle 12h 前提前自救
# MAX_RUN_SECONDS = 600   # Kaggle 12h 前提前自救


def save_checkpoint(path, model, optimizer, count,
                    x_train_this_epoch, x_test_left_epoch,
                    y_train_this_epoch, y_test_left_labels,
                    y_train_detection,
                    first_round_losses, online_losses, online_metrics,
                    run_dir, config=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    checkpoint = {
        "count": count,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "x_train_this_epoch": x_train_this_epoch.detach().cpu(),
        "x_test_left_epoch": x_test_left_epoch.detach().cpu(),
        "y_train_this_epoch": y_train_this_epoch.detach().cpu(),
        "y_test_left_labels": y_test_left_labels.detach().cpu(),
        "y_train_detection": y_train_detection.detach().cpu(),
        "first_round_losses": first_round_losses,
        "online_losses": online_losses,
        "online_metrics": online_metrics,
        "run_dir": run_dir,
        "config": config or {},
    }
    torch.save(checkpoint, path)


def load_checkpoint(path, model, optimizer, device):
    checkpoint = torch.load(path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    restored = {
        "count": checkpoint.get("count", 0),
        "x_train_this_epoch": checkpoint["x_train_this_epoch"].to(device),
        "x_test_left_epoch": checkpoint["x_test_left_epoch"].to(device),
        "y_train_this_epoch": checkpoint["y_train_this_epoch"].to(device),
        "y_test_left_labels": checkpoint["y_test_left_labels"].to(device),
        "y_train_detection": checkpoint["y_train_detection"].to(device),
        "first_round_losses": checkpoint.get("first_round_losses", []),
        "online_losses": checkpoint.get("online_losses", []),
        "online_metrics": checkpoint.get("online_metrics", {}),
        "run_dir": checkpoint.get("run_dir", os.path.dirname(path)),
        "config": checkpoint.get("config", {}),
    }
    return restored


def get_normal_templates(model, normal_data):
    with torch.no_grad():
        enc, dec = model(normal_data)
        normal_temp = torch.mean(F.normalize(enc, p=2, dim=1), dim=0)
        normal_recon_temp = torch.mean(F.normalize(dec, p=2, dim=1), dim=0)
    return normal_temp, normal_recon_temp


def main():
    parser = argparse.ArgumentParser(description="Online training with Kaggle resume support")
    parser.add_argument("--dataset", type=str, default="nsl")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--epoch_1", type=int, default=1)
    parser.add_argument("--percent", type=float, default=0.8)
    parser.add_argument("--flip_percent", type=float, default=0.2)
    parser.add_argument("--sample_interval", type=int, default=2000)
    parser.add_argument("--cuda", type=str, default="0")

    # BGMM parameters
    parser.add_argument("--bgmm_components", type=int, default=10)
    parser.add_argument("--bgmm_reg_covar", type=float, default=1e-4)
    parser.add_argument("--bgmm_max_fit_samples", type=int, default=5000)

    # Resume / checkpoint
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint file to resume from")
    parser.add_argument("--save_interval", type=int, default=50,
                        help="Save checkpoint every X online steps")

    args = parser.parse_args()

    dataset = args.dataset
    epochs = args.epochs
    epoch_1 = args.epoch_1
    percent = args.percent
    flip_percent = args.flip_percent
    sample_interval = args.sample_interval
    cuda_num = args.cuda

    bgmm_components = args.bgmm_components
    bgmm_reg_covar = args.bgmm_reg_covar
    bgmm_max_fit_samples = args.bgmm_max_fit_samples

    resume_path = args.resume
    save_interval = args.save_interval

    tem = 0.02
    bs = 128
    seed = 5009
    seed_round = 1

    device = torch.device(f"cuda:{cuda_num}" if torch.cuda.is_available() else "cpu")
    criterion = CRCLoss(device, tem)

    # -------------------- Load dataset --------------------
    if dataset == "nsl":
        input_dim = 121
        train_path = "NSL_pre_data/PKDDTrain+.csv"
        test_path = "NSL_pre_data/PKDDTest+.csv"

        train_df = load_data(train_path)
        test_df = load_data(test_path)

        splitter = SplitData(dataset="nsl")
        x_train, y_train = splitter.transform(train_df, labels="labels2")
        x_test, y_test = splitter.transform(test_df, labels="labels2")

    elif dataset == "unsw":
        input_dim = 196
        train_path = "UNSW_pre_data/UNSWTrain.csv"
        test_path = "UNSW_pre_data/UNSWTest.csv"

        train_df = load_data(train_path)
        test_df = load_data(test_path)

        splitter = SplitData(dataset="unsw")
        x_train, y_train = splitter.transform(train_df, labels="label")
        x_test, y_test = splitter.transform(test_df, labels="label")

    elif dataset == "cic":
        input_dim = None
        train_path = "/kaggle/input/datasets/chenghinchan/cic-pre-data/CICTrain.csv"
        test_path = "/kaggle/input/datasets/chenghinchan/cic-pre-data/CICTest.csv"

        train_df = load_data(train_path)
        test_df = load_data(test_path)

        splitter = SplitData(dataset="cic")
        x_train, y_train = splitter.transform(train_df, labels="label")
        x_test, y_test = splitter.transform(test_df, labels="label")

        input_dim = x_train.shape[1]
        print(f"CIC-IDS-2017 input_dim = {input_dim}")
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    x_train = torch.FloatTensor(x_train)
    y_train = torch.LongTensor(y_train)
    x_test = torch.FloatTensor(x_test)
    y_test = torch.LongTensor(y_test)

    # -------------------- Main loop --------------------
    for i in range(seed_round):
        current_seed = seed + i
        setup_seed(current_seed)

        # split first, so resume and non-resume keep the same base split
        online_x_train, online_x_test, online_y_train, online_y_test = train_test_split(
            x_train, y_train, test_size=percent, random_state=current_seed
        )
        num_of_first_train = online_x_train.shape[0]

        train_ds = TensorDataset(online_x_train, online_y_train)
        train_loader = torch.utils.data.DataLoader(
            dataset=train_ds, batch_size=bs, shuffle=True
        )

        model = AE(input_dim).to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

        # Move commonly used tensors to device
        x_train_device = x_train.to(device)
        x_test_device = x_test.to(device)
        y_test_device = y_test.to(device)
        online_x_train_device = online_x_train.to(device)
        online_y_train_device = online_y_train.to(device)

        start_count = 0
        skip_stage1 = False
        finished_normally = True
        first_round_losses = []
        online_losses = []
        online_metrics = {}

        # Decide run_dir
        if resume_path is not None and os.path.exists(resume_path):
            restored = load_checkpoint(resume_path, model, optimizer, device)

            start_count = restored["count"]
            x_train_this_epoch = restored["x_train_this_epoch"]
            x_test_left_epoch = restored["x_test_left_epoch"]
            y_train_this_epoch = restored["y_train_this_epoch"]
            y_test_left_labels = restored["y_test_left_labels"]
            y_train_detection = restored["y_train_detection"]
            first_round_losses = restored["first_round_losses"]
            online_losses = restored["online_losses"]
            online_metrics = restored["online_metrics"]

            run_dir = restored["run_dir"]
            os.makedirs(run_dir, exist_ok=True)

            skip_stage1 = True
            print(f"[*] 检测到 checkpoint，正在从 {resume_path} 恢复...")
            print(f"[*] 恢复成功，将从 Stage2 的 step={start_count} 继续运行")
            timestamp = os.path.basename(run_dir).split("_")[-1] if "_" in os.path.basename(run_dir) else datetime.now().strftime("%Y%m%d_%H%M%S")
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_dir = os.path.join("result", f"{dataset}_seed{current_seed}_{timestamp}")
            os.makedirs(run_dir, exist_ok=True)

        ckpt_path = os.path.join(run_dir, "latest_checkpoint.pth")

        checkpoint_config = {
            "dataset": dataset,
            "seed": current_seed,
            "epochs": epochs,
            "epoch_1": epoch_1,
            "percent": percent,
            "flip_percent": flip_percent,
            "sample_interval": sample_interval,
            "batch_size": bs,
            "temperature": tem,
            "input_dim": input_dim,
            "num_first_train": int(num_of_first_train),
            "timestamp": timestamp,
            "bgmm_components": bgmm_components,
            "bgmm_reg_covar": bgmm_reg_covar,
            "bgmm_max_fit_samples": bgmm_max_fit_samples,
        }

        # -------------------- Stage 1: Offline Training --------------------
        if not skip_stage1:
            model.train()
            for epoch in range(epochs):
                epoch_loss = 0.0
                num_batches = 0

                for inputs, labels in train_loader:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    optimizer.zero_grad()
                    features, recon_vec = model(inputs)
                    loss = criterion(features, labels) + criterion(recon_vec, labels)
                    loss.backward()
                    optimizer.step()

                    epoch_loss += loss.item()
                    num_batches += 1

                avg_loss = epoch_loss / max(num_batches, 1)
                first_round_losses.append(avg_loss)
                print(f"[Stage1] seed={current_seed}, epoch={epoch+1}/{epochs}, loss={avg_loss:.6f}")

            x_train_this_epoch = online_x_train_device.clone()
            x_test_left_epoch = online_x_test.to(device).clone()
            y_train_this_epoch = online_y_train_device.clone()
            y_test_left_labels = online_y_test.to(device).clone()
            y_train_detection = y_train_this_epoch.clone()

        # -------------------- Stage 2: Online Training --------------------
        count = start_count

        if not skip_stage1:
            total_online_samples = len(x_test_left_epoch)
        else:
            total_online_samples = count * sample_interval + len(x_test_left_epoch)

        total_online_steps = (total_online_samples + sample_interval - 1) // sample_interval

        try:
            while len(x_test_left_epoch) > 0:
                # Kaggle time protection
                if time.time() - START_TIME > MAX_RUN_SECONDS:
                    print("\n[!] 已运行接近 12 小时，开始自动存档并安全退出...")
                    save_checkpoint(
                        ckpt_path, model, optimizer, count,
                        x_train_this_epoch, x_test_left_epoch,
                        y_train_this_epoch, y_test_left_labels,
                        y_train_detection,
                        first_round_losses, online_losses, online_metrics,
                        run_dir, checkpoint_config
                    )
                    print(f"[*] 超时存档已保存到: {ckpt_path}")

                    shutil.make_archive(run_dir, "zip", run_dir)
                    print(f"[*] 结果目录已打包: {run_dir}.zip")

                    finished_normally = False
                    break

                count += 1
                processed = min(count * sample_interval, total_online_samples)

                if len(x_test_left_epoch) < sample_interval:
                    x_test_this_epoch = x_test_left_epoch.clone()
                    y_true_this_step = y_test_left_labels.clone()
                    x_test_left_epoch = x_test_left_epoch[:0]
                    y_test_left_labels = y_test_left_labels[:0]
                else:
                    x_test_this_epoch = x_test_left_epoch[:sample_interval].clone()
                    y_true_this_step = y_test_left_labels[:sample_interval].clone()
                    x_test_left_epoch = x_test_left_epoch[sample_interval:]
                    y_test_left_labels = y_test_left_labels[sample_interval:]

                # use original clean normal samples to build templates
                normal_data = online_x_train_device[(online_y_train_device == 0).squeeze()]
                normal_temp, normal_recon_temp = get_normal_templates(model, normal_data)

                predict_label = evaluate(
                    normal_temp,
                    normal_recon_temp,
                    x_train_this_epoch,
                    y_train_detection,
                    x_test_this_epoch,
                    0,
                    model,
                    n_components=bgmm_components,
                    reg_covar=bgmm_reg_covar,
                    random_state=current_seed,
                    max_fit_samples=bgmm_max_fit_samples
                )

                if isinstance(predict_label, torch.Tensor):
                    y_pred_step = predict_label.clone().detach().to(device)
                    y_pred_np = y_pred_step.cpu().numpy()
                else:
                    y_pred_np = np.array(predict_label)
                    y_pred_step = torch.LongTensor(y_pred_np).to(device)

                y_true_np = y_true_this_step.detach().cpu().numpy()

                batch_acc = accuracy_score(y_true_np, y_pred_np)
                batch_prec = precision_score(y_true_np, y_pred_np, zero_division=0)
                batch_rec = recall_score(y_true_np, y_pred_np, zero_division=0)
                batch_f1 = f1_score(y_true_np, y_pred_np, zero_division=0)
                online_metrics[count] = (batch_acc, batch_prec, batch_rec, batch_f1)

                print(
                    f"[Stage2] seed={current_seed}, step={count}/{total_online_steps} "
                    f"({100 * processed / total_online_samples:.1f}%) | "
                    f"Acc={batch_acc:.4f}  Prec={batch_prec:.4f}  "
                    f"Rec={batch_rec:.4f}  F1={batch_f1:.4f}"
                )

                # detection labels (unflipped)
                y_train_detection = torch.cat((y_train_detection, y_pred_step), dim=0)

                # flipped pseudo labels for training
                y_train_pseudo = y_pred_step.clone()
                num_flip = int(flip_percent * y_train_pseudo.shape[0])
                if num_flip > 0:
                    flip_indices = np.random.choice(y_train_pseudo.shape[0], num_flip, replace=False)
                    flip_indices = torch.LongTensor(flip_indices).to(device)
                    y_train_pseudo[flip_indices] = 1 - y_train_pseudo[flip_indices]

                x_train_this_epoch = torch.cat((x_train_this_epoch, x_test_this_epoch), dim=0)
                y_train_this_epoch = torch.cat((y_train_this_epoch, y_train_pseudo), dim=0)

                train_ds = TensorDataset(x_train_this_epoch, y_train_this_epoch)
                train_loader = torch.utils.data.DataLoader(
                    dataset=train_ds, batch_size=bs, shuffle=True
                )

                model.train()
                step_loss = 0.0
                step_batches = 0

                for _ in range(epoch_1):
                    for inputs, labels in train_loader:
                        inputs = inputs.to(device)
                        labels = labels.to(device)

                        optimizer.zero_grad()
                        features, recon_vec = model(inputs)
                        loss = criterion(features, labels) + criterion(recon_vec, labels)
                        loss.backward()
                        optimizer.step()

                        step_loss += loss.item()
                        step_batches += 1

                online_losses.append(step_loss / max(step_batches, 1))

                if count % save_interval == 0:
                    save_checkpoint(
                        ckpt_path, model, optimizer, count,
                        x_train_this_epoch, x_test_left_epoch,
                        y_train_this_epoch, y_test_left_labels,
                        y_train_detection,
                        first_round_losses, online_losses, online_metrics,
                        run_dir, checkpoint_config
                    )
                    print(f"[*] 自动存档: {ckpt_path} (step={count})")

        except KeyboardInterrupt:
            print(f"\n[!] 检测到手动中断，正在紧急保存当前进度 (step={count})...")
            save_checkpoint(
                ckpt_path, model, optimizer, count,
                x_train_this_epoch, x_test_left_epoch,
                y_train_this_epoch, y_test_left_labels,
                y_train_detection,
                first_round_losses, online_losses, online_metrics,
                run_dir, checkpoint_config
            )
            print(f"[*] 紧急存档已保存: {ckpt_path}")
            finished_normally = False

        # if interrupted / timed out, skip final evaluation
        if not finished_normally:
            print("[*] 本次运行已安全存档并提前结束，不执行最终评估。下次用 --resume 继续。")
            continue

        # -------------------- Final Evaluation --------------------
        normal_data = online_x_train_device[(online_y_train_device == 0).squeeze()]
        normal_temp, normal_recon_temp = get_normal_templates(model, normal_data)

        res_en, res_de, res_final, y_pred_final = evaluate(
            normal_temp,
            normal_recon_temp,
            x_train_this_epoch,
            y_train_detection,
            x_test_device,
            y_test_device,
            model,
            return_predictions=True,
            n_components=bgmm_components,
            reg_covar=bgmm_reg_covar,
            random_state=current_seed,
            max_fit_samples=bgmm_max_fit_samples
        )

        print(f'\n{"=" * 60}')
        print(f"  Final Results - {dataset.upper()} seed={current_seed}")
        print(f'{"=" * 60}')
        print(f'  {"":12s} {"Acc":>8s} {"Prec":>8s} {"Recall":>8s} {"F1":>8s}')
        print(f'  {"-" * 44}')
        print(f'  {"Encoder":12s} {res_en[0]:>8.4f} {res_en[1]:>8.4f} {res_en[2]:>8.4f} {res_en[3]:>8.4f}')
        print(f'  {"Decoder":12s} {res_de[0]:>8.4f} {res_de[1]:>8.4f} {res_de[2]:>8.4f} {res_de[3]:>8.4f}')
        print(f'  {"Combined":12s} {res_final[0]:>8.4f} {res_final[1]:>8.4f} {res_final[2]:>8.4f} {res_final[3]:>8.4f}')
        print(f'{"=" * 60}')

        # -------------------- Save results --------------------
        metric_names = ["accuracy", "precision", "recall", "f1"]
        run_result = {
            "config": checkpoint_config,
            "stage1_losses": first_round_losses,
            "stage2_losses": online_losses,
            "online_metrics": {
                str(step): dict(zip(metric_names, [float(v) for v in vals]))
                for step, vals in online_metrics.items()
            },
            "final_results": {
                "encoder": dict(zip(metric_names, [float(v) for v in res_en])),
                "decoder": dict(zip(metric_names, [float(v) for v in res_de])),
                "combined": dict(zip(metric_names, [float(v) for v in res_final])),
            },
        }

        with open(os.path.join(run_dir, "metrics.json"), "w", encoding="utf-8") as f:
            json.dump(run_result, f, indent=2)

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            },
            os.path.join(run_dir, "model.pth")
        )

        y_test_np = y_test_device.detach().cpu().numpy()
        y_pred_final_np = y_pred_final.detach().cpu().numpy() if isinstance(y_pred_final, torch.Tensor) else np.array(y_pred_final)

        np.savez(
            os.path.join(run_dir, "predictions.npz"),
            y_true=y_test_np,
            y_pred=y_pred_final_np,
        )

        plot_training_summary(
            first_round_losses=first_round_losses,
            online_losses=online_losses,
            online_metrics=online_metrics,
            final_encoder=res_en,
            final_decoder=res_de,
            final_combined=res_final,
            y_test_true=y_test_np,
            y_test_pred=y_pred_final_np,
            dataset=dataset,
            seed=current_seed,
            save_dir=run_dir,
        )

        # save final zip
        shutil.make_archive(run_dir, "zip", run_dir)
        print(f"  [Saved] All results -> {run_dir}/")
        print(f"  [Saved] Zip archive -> {run_dir}.zip")


if __name__ == "__main__":
    main()
