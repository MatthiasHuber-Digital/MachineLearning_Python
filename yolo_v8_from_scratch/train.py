import config
import torch
import torch.optim as optim

from model import YOLOv3
from tqdm import tqdm
from utils import (
    mean_average_precision,
    cells_to_boxes,
    get_evaluation_bboxes,
    save_checkpoint,
    load_checkpoint,
    check_class_accuracy,
    get_loaders,
    plot_couple_examples,
)
from loss import YoloLoss

torch.backends.cudnn.benchmark = True # performance improvement via cuda torch backend


def combined_loss(
    prediction: list, 
    ground_truth:list, 
    scaled_anchors: list):
    return (
        loss_fn(prediction[0], ground_truth[0], scaled_anchors[0])
        + loss_fn(prediction[1], ground_truth[1], scaled_anchors[1])
        + loss_fn(prediction[2], ground_truth[2], scaled_anchors[2])
    )


# using MIXED PRECISION TRAINING
def train_fn(
    train_loader,
    model,
    optimizer, 
    scaler,
    scaled_anchors,
    ): # 1 epoch of training
    loop = tqdm(train_loader, leave=True)
    losses = []
    
    for batch_idx, (x,y) in enumerate(loop):
        x = x.to(config.DEVICE)
        y0, y1, y2 = (
            y[0].to(config.DEVICE),
            y[1].to(config.DEVICE),
            y[2].to(config.DEVICE),
        )
        y = list(y0,y1,y2)

        #forward pass:
        with torch.cuda.amp.autocast(): # float16 in pytorch - reduces VRAM -50% approximately
            out = model(x)
            loss = combined_loss(
                prediction=out,
                ground_truth=y,
                scaled_anchors=scaled_anchors,
            )

        losses.append(loss.item())
        
        #backprop of gradients
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        mean_loss = sum(losses) / len(losses)
        loop.set_postfix(loss=mean_loss) # progress bar update


def main():
    model = YOLOv3(amount_of_classes=config.NUM_CLASSES).to(config.DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    loss_func = YoloLoss()
    scaler = torch.cuda.amp.grad_scaler()
        
    train_loader, test_loader, train_eval_loader  = get_loaders(
        train_csv_path=config.DATASET+"/8examples.csv",
        test_csv_path=config.DATASET+"/8examples.csv",
    )
    
    if config.LOAD_MODEL:
        load_checkpoint(
            checkpoint_file=config.CHECKPOINT_FILE,
            model=model,
            optimizer=optimizer,
            lr=config.LEARNING_RATE,
        )
    
    scaled_anchors = (
        torch.tensor(config.ANCHORS)
        *
        torch.tensor(config.S).unsqueeze(1).unsqueeze(2).repeat(1,3,2)
    ).to(config.DEVICE)
    
    for epoch in range(config.NUM_EPOCHS):
        train_fn(train_loader=test_loader, # train_loader 
                 model=model,
                 optimizer=optimizer,
                 scaler=scaler,
                 loss_func=loss_func,
                 scaled_anchors=scaled_anchors,
                 )

        if config.SAVE_MODEL:
            save_checkpoint(model=model, optimizer=optimizer)
        
        if epoch % 10 == 0 and epoch > 0:
            print("-- test accuracy is:")
            check_class_accuracy(
                model=model,
                loader=test_loader,
                threshold=config.CONF_THRESHOLD,
            )

            pred_boxes, true_boxes = get_evaluation_bboxes(
                loader=test_loader,
                model=model,
                iou_threshold=config.NMS_IOU_THRESH,
                anchors=config.ANCHORS,
                threshold=config.CONF_THRESHOLD,
            )
            mAP_val_data = mean_average_precision(
                pred_boxes=pred_boxes,
                true_boxes=true_boxes,
                iou_threshold=config.MAP_IOU_THRESH,
                box_format="midpoint",
                num_classes=config.NUM_CLASSES,
            )
            print(f"-- test mean average precision: {mAP_val_data.item()}")


if __name__ == "__main__":
    main()
    