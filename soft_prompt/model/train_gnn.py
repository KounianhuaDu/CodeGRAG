import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os

sys.path.append("../")
from dataloaders.gnn_dataloader import CodeGNNDataset, load_data
from model.gnn import CodeGNN
import random
import numpy as np
import argparse
from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser()
    # dataset
    parser.add_argument(
        "--dataset", type=str, default="appsnew", choices=["appsnew", "CodeContest"]
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="/home/jzchen/ML/AfterCodegrag/raw_data/APPS/dgl",
    )
    parser.add_argument("--num_workers", type=int, default=8)
    # device
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    # training
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--wd", type=float, default=1e-5)
    parser.add_argument("--early_stop", type=int, default=10)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--alpha", type=float, default=1)
    parser.add_argument("--beta", type=float, default=0.5)
    # model structure
    parser.add_argument("--embedding_size", type=int, default=32)
    args = parser.parse_args()
    return args


def seed_all(seed, gpu):
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Embedding):
        nn.init.xavier_normal_(m.weight)


def add_gaussian_noise_torch(tensor, mean=0, std=1, device="cpu"):
    """
    向torch.tensor添加高斯噪声。

    参数:
        tensor (torch.Tensor): 输入张量。
        mean (float): 高斯噪声的平均值。
        std (float): 高斯噪声的标准差。

    返回:
        torch.Tensor: 添加了噪声的张量。
    """
    noise = torch.normal(mean, std, size=tensor.shape).to(device)
    return tensor + noise


def main(args):
    # set device and seed
    device = torch.device(
        f"cuda:{args.gpu}" if (torch.cuda.is_available() and args.gpu >= 0) else "cpu"
    )
    args.trained_model_path = f"../trained_models/{args.dataset}/CodeGNN/"
    os.makedirs(args.trained_model_path, exist_ok=True)

    print(f"Device is {device}.")
    seed_all(args.seed, device)

    # Dataloader
    train_loader = load_data(
        batch_size=args.batch_size,
        path=args.data_path,
    )
    # Model
    model = CodeGNN(256, 256).to(device)
    model.apply(weight_init)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

    # start training
    print("Start training.")
    best_loss = float("inf")
    kill_cnt = 0
    for epoch in range(args.epochs):
        train_loss = []
        model.train()
        with tqdm(total=len(train_loader), dynamic_ncols=True) as t:
            t.set_description(f"Epoch: {epoch+1}/{args.epochs}")
            for step, (batch_codes, batch_graphs) in enumerate(train_loader):
                batch_codes, batch_graphs = batch_codes.to(device), batch_graphs.to(
                    device
                )
                graph_emb = model.get_graph_embedding(batch_graphs)
                inter_loss = model.inter_contrast(batch_codes, graph_emb)
                intra_loss = model.intra_contrast(
                    graph_emb, add_gaussian_noise_torch(graph_emb, device=device)
                )
                tr_loss = args.alpha * inter_loss + args.beta * intra_loss
                train_loss.append(tr_loss.item())

                # backward
                optimizer.zero_grad()
                tr_loss.backward()
                optimizer.step()

                t.update()
                t.set_postfix({"Train loss": f"{tr_loss.item():.4f}"})

        train_loss = np.mean(train_loss)
        if train_loss < best_loss:
            best_auc = train_loss
            best_epoch = epoch
            torch.save(
                model.state_dict(),
                os.path.join(
                    args.trained_model_path,
                    f"CodeGNN_{args.lr}_{args.wd}_{args.alpha}_{args.beta}.pth",
                ),
            )
            kill_cnt = 0
            print("saving model...")
        else:
            kill_cnt += 1
            if kill_cnt >= args.early_stop:
                print(f"Early stop at epoch {epoch}")
                print("best epoch: {}".format(best_epoch + 1))
                break


if __name__ == "__main__":
    args = get_args()
    main(args)
