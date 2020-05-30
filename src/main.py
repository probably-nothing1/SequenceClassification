import argparse
import os
import time
import numpy as np

import wandb
import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from models import LSTM
from utils import parse_args, save_model
from dataset import get_train_dataloader, get_test_dataloader
from training import dispatch_optimizer, get_lr
from metrics import compute_accuracy

save_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'models')
embeddings = []

if __name__ == '__main__':
    args = parse_args()
    print(args)

    use_cuda = not args.use_cpu and torch.cuda.is_available()
    device = 'cuda' if use_cuda else 'cpu'
    bs = args.train_batch_size
    best_acc = 0

    train_dataloader = get_train_dataloader(args.data_dir, args.train_batch_size, embedding=args.embedding)
    test_dataloader = get_test_dataloader(args.data_dir, args.train_batch_size, embedding=args.embedding)

    # metrics_train_dataloader = None # et_train_dataloader(args.data_dir, eval_batch, dataset_version, shuffle=False, use_transforms=False)
    # metrics_test_dataloader = None # get_test_dataloader(args.data_dir, eval_batch, dataset_version, shuffle=False, use_transforms=False)

    model = LSTM(args.num_layers, args.hidden_size, embedding=args.embedding).to(device)

    wandb.init(project=args.project_name, name=args.run_name, config=args)
    wandb.watch(model, log='all')
    config = wandb.config

    loss_function = CrossEntropyLoss(reduction='mean')
    optimizer = dispatch_optimizer(model, args)
    # lr_scheduler = dispatch_lr_scheduler(optimizer, args)

    iteration = 0
    training_accuracy = compute_accuracy(model, train_dataloader, device)
    test_accuracy = compute_accuracy(model, test_dataloader, device)
    wandb.log({'training accuracy': training_accuracy}, step=iteration*bs)
    wandb.log({'test_accuracy': test_accuracy}, step=iteration * bs)

    for epoch in range(args.epochs):
        print(f'epoch {epoch}')
        for x, y in train_dataloader:
            start_time = time.time()
            if args.test_random_truncate:
                if np.random.rand() < 0.25:
                    x = x[:, :4]
                elif np.random.rand() < 0.33:
                    x = x[:, :3]
                # elif np.random.rand() < 0.50:
                #     x = x[:, :2]
            x, y = x.to(device), y.to(device)
            class_probabilities = model(x)

            loss = loss_function(class_probabilities, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            wandb.log({'training loss': loss}, step=iteration*bs)
            wandb.log({'learning rate': get_lr(optimizer)}, step=iteration*bs)

            wandb.log({'iteration': iteration}, step=iteration*bs)
            wandb.log({'iteration time': (time.time() - start_time) / bs}, step=iteration*bs)
            # if iteration % 10 == 0:
            #     test_loss = compute_loss(model, test_dataloader, loss_function, device)
            #     wandb.log({'test loss': loss}, step=iteration*bs)
            iteration += 1

        # lr_scheduler.step()
        training_accuracy = compute_accuracy(model, train_dataloader, device)
        test_accuracy = compute_accuracy(model, test_dataloader, device)
        wandb.log({'training accuracy': training_accuracy}, step=iteration*bs)
        wandb.log({'test_accuracy': test_accuracy}, step=iteration*bs)
        if test_accuracy > best_acc:
            best_acc = test_accuracy
        print(test_accuracy)
        save_model(model, f'{save_dir}/{args.run_name}_{epoch}.pt')

        if args.embedding:
            points = torch.arange(0, 100, dtype=torch.long).cuda()
            embedding = model.compute_embeddings(points).cpu().detach().numpy()
            embeddings.append(embedding)

    np.save(f'{save_dir}/{args.run_name}_{args.epochs}_embeddings.npy', embeddings)