import torchvision
from .resnet import resnet

def load_model(args, n_classes):

    if args.architecture == "resnet18":
        model = resnet("resnet18", num_classes=n_classes)

    elif args.architecture == "resnet34":
        model = torchvision.models.resnet34(pretrained=False, progress=True)

    elif args.architecture == "resnet50":
        model = torchvision.models.resnet50(pretrained=False, progress=True)

    elif args.architecture == "resnet101":
        model = torchvision.models.resnet101(pretrained=False, progress=True)

    else: 
        raise ValueError(f"{args.architecture} not recognized.")

    return model
