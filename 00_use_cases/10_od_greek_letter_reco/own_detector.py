import torch.utils
import torch.utils.data
import torchvision
from torchvision import transforms as T
import torch
from torch.utils.data import default_collate
from custom_dataset import CustomDataset
from engine import train_one_epoch, evaluate


def get_transforms(train: bool):

    transforms = []
    if train:
        transforms.append(T.RandomRotation(degrees=180)) # other transformations
    transforms.append(T.v2.ToDtype(torch.float))
    transforms.append(T.ToTensor())

    return T.Compose(transforms)


def main():
    model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(weights="DEFAULT")

    in_features = model.roi_heads.box_predictor.cls_score.in_features

    num_classes = 4

    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.MobileNet_V3_Large_Weights(in_features, num_classes)

    my_transforms = get_transforms()

    data_train = CustomDataset(root='data', transforms=my_transforms)

    data_loader = torch.utils.data.DataLoader(
        dataset=data_train,
        batch_size=8,
        shuffle=True,
    )
    
    device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'

    # move model to the right device
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params,
        lr=0.005,
        momentum=0.9,
        weight_decay=0.0005
    )

    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=3,
        gamma=0.1
    )

    # let's train it just for 2 epochs
    num_epochs = 2

    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        #evaluate(model, data_loader_test, device=device)

    print("That's it!")


if __name__ == "__main__":
    main()