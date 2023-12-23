import argparse

def get_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--epoch", default=5, type=int)
    
    parser.add_argument("--backbone_model", default="dinov2", choices=["resnet50", "dinov2"], type=str)
    parser.add_argument("--split", default="1", type=str)
    parser.add_argument("--output_folder", default="outputs", type=str)

    args = parser.parse_args()

    if args.backbone_model == "dinov2":
        args.cls_model_name = "cls_dinov2.pth"
        args.fc_dim = 384
    elif args.backbone_model == "resnet50":
        args.cls_model_name = "cls_resnet50.pth"
        args.fc_dim = 2048

    return args