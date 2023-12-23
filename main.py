import os.path as osp

import torch
import torch.nn as nn
import torch.optim as optim

from lib.models import load_backbone, load_classifier
from lib.datasets import get_dataloader
from lib.utils import AverageMeter, cal_acc
from config import get_args

def main(args):
    model_save_root = osp.join(args.output_folder, args.cls_model_name)

    backbone = load_backbone(model=args.backbone_model)
    classifier = load_classifier(args.fc_dim)

    optimizer = optim.AdamW(classifier.parameters(), lr=1e-3)
    ce_loss = nn.CrossEntropyLoss()
    
    train_dataloader = get_dataloader(batch_size=64, shuffle=True, num_workers=4, mode="train", split=args.split, model=args.backbone_model)
    valid_dataloader = get_dataloader(batch_size=64, shuffle=False, num_workers=4, mode="valid", split=args.split, model=args.backbone_model)

    best_acc = 0
    for epoch in range(args.epoch):
        train_losses = AverageMeter()
        train_acces = AverageMeter()
        for imgs, labels in train_dataloader:
            imgs = imgs.cuda()
            labels = labels.cuda()
            bs = imgs.shape[0]

            with torch.no_grad():
                embeddings = backbone(imgs)
            outputs = classifier(embeddings)
            loss = ce_loss(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.update(loss.item(), bs)
            acc = cal_acc(outputs, labels)
            train_acces.update(acc, bs)

        valid_losses = AverageMeter()
        valid_acces = AverageMeter()
        with torch.no_grad():
            for imgs, labels in valid_dataloader:
                imgs = imgs.cuda()
                labels = labels.cuda()
                bs = imgs.shape[0]

                embeddings = backbone(imgs)
                outputs = classifier(embeddings)
                loss = ce_loss(outputs, labels)
                valid_losses.update(loss.item(), bs)
                acc = cal_acc(outputs, labels)
                valid_acces.update(acc, bs)

        print(f"Epoch: {epoch+1} | train loss: {train_losses.avg:.4f} | train acc: {train_acces.avg*100:.1f} | valid loss: {valid_losses.avg:.4f} | valid acc: {valid_acces.avg*100:.1f} ")

        if valid_acces.avg > best_acc:
            best_acc = valid_acces.avg
            torch.save(classifier.state_dict(), model_save_root)
            print(f"Save best model at epoch {epoch+1}.")
    
    classifier = load_classifier(args.fc_dim, model_save_root)

    test_dataloader = get_dataloader(batch_size=64, shuffle=False, num_workers=4, mode="train", split=args.split, model=args.backbone_model)

    test_losses = AverageMeter()
    test_acces = AverageMeter()
    with torch.no_grad():
        for imgs, labels in test_dataloader:
            imgs = imgs.cuda()
            labels = labels.cuda()
            bs = imgs.shape[0]

            embeddings = backbone(imgs)
            outputs = classifier(embeddings)
            loss = ce_loss(outputs, labels)
            test_losses.update(loss.item(), bs)
            acc = cal_acc(outputs, labels)
            test_acces.update(acc, bs)
    
    print(f"Test loss: {test_losses.avg:.4f} | test acc: {test_acces.avg*100:.1f}")


if __name__ == "__main__":
    args = get_args()
    main(args)