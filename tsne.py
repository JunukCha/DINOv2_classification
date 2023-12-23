import os.path as osp

from sklearn.manifold import TSNE

import numpy as np
import torch
import matplotlib.pyplot as plt

from lib.models import load_backbone
from lib.datasets import get_dataloader
from config import get_args

def main(args):
    backbone = load_backbone(model=args.backbone_model)

    test_dataloader = get_dataloader(batch_size=64, shuffle=False, num_workers=4, mode="test", split=args.split, model=args.backbone_model)

    tsne = TSNE(n_components=2, random_state=42)

    test_images = []
    test_embbedings = []
    test_labels = []
    with torch.no_grad():
        for imgs, labels in test_dataloader:
            imgs = imgs.cuda()
            labels = labels.cuda()
            embeddings = backbone(imgs)
            imgs_np = ((imgs.cpu().numpy()*0.5+0.5)*255).transpose(0, 2, 3, 1)
            imgs_np = imgs_np.astype(np.uint8)
            test_images.append(imgs_np)
            test_embbedings.append(embeddings.cpu().numpy())
            test_labels.append(labels.cpu().numpy())
    
    test_images = np.concatenate(test_images)
    test_embbedings = np.concatenate(test_embbedings)
    test_labels = np.concatenate(test_labels)
    X_2d = tsne.fit_transform(test_embbedings)
    cmap = plt.cm.get_cmap('tab20', 20)  

    fig, ax_scatter = plt.subplots(1, 1, figsize=(10, 8))
    
    for i in range(17):
        indices = test_labels == i
        ax_scatter.scatter(X_2d[indices, 0], X_2d[indices, 1], c=cmap(i), label=f"Class {i}")
    ax_scatter.set_xlabel('t-SNE feature 1')
    ax_scatter.set_ylabel('t-SNE feature 2')
    ax_scatter.set_title(f't-SNE visualization {args.backbone_model}')
    ax_scatter.legend()
    plt.savefig(osp.join(args.output_folder, f"tsne_{args.backbone_model}.jpg"))

if __name__ == "__main__":
    args = get_args()
    main(args)