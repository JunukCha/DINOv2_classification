from sklearn.manifold import TSNE

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

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

    fig, (ax_scatter, ax_image) = plt.subplots(1, 2, figsize=(12, 6))
    
    scatter = ax_scatter.scatter(X_2d[:, 0], X_2d[:, 1], s=10, c=test_labels, cmap=cmap)
    ax_scatter.set_xlabel('t-SNE feature 1')
    ax_scatter.set_ylabel('t-SNE feature 2')
    ax_scatter.set_title(f't-SNE visualization {args.backbone_model}')

    ax_image.imshow(np.ones((224, 224, 3)), aspect="auto")
    ax_image.axis('off')
    
    circle = ax_scatter.scatter(0, 0, s=10, c="k")
    circle.set_visible(False)

    def hover(event):
        is_contained, annotation_index = scatter.contains(event)
        if is_contained:
            data_index = annotation_index['ind'][0]
            ax_image.clear()
            ax_image.imshow(test_images[data_index], aspect='auto')
            ax_image.axis('off')

            x, y = X_2d[data_index]
            circle.set_offsets([x, y])
            circle.set_visible(True)

            fig.canvas.draw_idle()
        else:
            ax_image.imshow(np.ones((224, 224, 3)), aspect='auto')
            ax_image.axis('off')

            circle.set_visible(False)

            fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", hover)
    plt.show()

if __name__ == "__main__":
    args = get_args()
    main(args)