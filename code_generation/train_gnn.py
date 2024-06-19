import argparse
import random

import dgl
from dgl import DropEdge
import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
from tqdm import tqdm
from tensorboardX import SummaryWriter

import os
import sys
sys.path.append("..")

from algo.Search_with_GNN import CodeGNN

from utils.gnn_dataloader import load_data


def main(args):
    # random seed & log file
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)

    log_file = open(os.path.join(args.output, 'gnn.txt'), "w+")
    train_dir = os.path.join(args.output, "train")
    train_writer = SummaryWriter(log_dir=train_dir)

    # step 1: Check device
    if args.gpu >= 0 and torch.cuda.is_available():
        device = "cuda:{}".format(args.gpu)
    else:
        device = "cpu"

    # step 2: Load data
    train_loader = load_data(
        args.batch_size, args.data, args.num_workers
    )
    print("Data loaded.")
    log_file.write("Data loaded.\n")

    # step 3: Create model and training components
    model = CodeGNN(
        in_feat=args.in_size,
        hidden_feat=args.hidden_size
    )
    model = model.to(device)
    transform = DropEdge(p=args.drop_edge)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    decay = optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_decay)
    print("Model created.")
    log_file.write("Model created.\n")

    # step 4: Training
    print("Start training.")
    log_file.write("Start training.\n")
    
    kill_cnt = 0
    for epoch in range(args.epochs):
        # Training and validation
        model.train()
        train_losses = []
        inter_losses = []
        intra_losses = []
        
        with tqdm(total=len(train_loader), dynamic_ncols=True) as t:
            for step, batch in enumerate(train_loader):
                code, g = batch
                corrupted_g = transform(g)
                code = code.to(device)
                g = g.to(device)
                corrupted_g = corrupted_g.to(device)
                
                graph_vec = model.get_graph_embedding(g)
                graph_vec_pos = model.get_graph_embedding(corrupted_g)

                inter_loss = model.inter_contrast(code, graph_vec)
                intra_loss = model.intra_contrast(graph_vec, graph_vec_pos)
                loss = inter_loss + args.alpha*intra_loss
                
                # compute loss
                train_losses.append(loss.item())
                inter_losses.append(inter_loss.item())
                intra_losses.append(intra_loss.item())
                
                # backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                t.update()
                t.set_description(desc=f'Epoch: {epoch}/{args.epochs}')
                t.set_postfix({
                    'train loss': f'{loss.item():.4f}',
                    'inter loss': f'{inter_loss.item():.4f}',
                    'intra loss': f'{intra_loss.item():.4f}'
                })

                train_iter = epoch * len(train_loader) + step
                if step % 10 == 0:
                    train_writer.add_scalar("train_loss", loss.item(), train_iter)
                    train_writer.add_scalar("inter_loss", inter_loss.item(), train_iter)
                    train_writer.add_scalar("intra_loss", intra_loss.item(), train_iter)
                if step % 100 == 0:
                    log_file.write("Epoch {}, step {}/{}, train loss: {:.4f}, inter loss: {:.4f}, intra loss: {:.4f}\n".format(epoch, step, len(train_loader), loss.item(), inter_loss.item(), intra_loss.item()))

        train_losses = np.mean(train_losses)
        inter_losses = np.mean(inter_losses)
        intra_losses = np.mean(intra_losses)

        log_file.write("Epoch {}, train loss: {}, inter loss: {:.4f}, intra loss: {:.4f}\n".format(epoch, train_losses, inter_losses, intra_losses))
        torch.save(model.state_dict(), os.path.join(args.output, f'model_dict_{epoch}'))
        
        decay.step()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Parser For Arguments", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--data", default="../data", help="Path to save the data")
    parser.add_argument("--output", default="../output", help="Path to save the output")

    parser.add_argument("--in_size", type=int, default=256, help="Initial dimension size for entities.")
    parser.add_argument('--hidden_size', type=int, default=256, help='Hidden dimension size.')

    parser.add_argument("--gpu", type=int, default="0", help="Set GPU Ids : Eg: For CPU = -1, For Single GPU = 0")
    parser.add_argument("--epochs", type=int, default=20, help="Maximum number of epochs")

    parser.add_argument("--batch_size", type=int, default=100, help="Batch size.")
    parser.add_argument("--wd", type=float, default=5e-4, help="L2 Regularization for Optimizer")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning Rate")
    parser.add_argument("--lr_decay", type=float, default=1, help="Exponential decay of learning rate")
    parser.add_argument("--drop_edge", type=float, default=0.2)
    parser.add_argument("--alpha", type=float, default=1)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--num_workers", type=int, default=10, help="Number of processes to construct batches")
    parser.add_argument("--early_stop", default=3, type=int, help="Patience for early stop.")

    parser.add_argument("--dropout", default=0.1, type=float, help="Dropout.")

    args = parser.parse_args()

    print(args)
    os.makedirs(args.output, exist_ok=True)

    main(args)
