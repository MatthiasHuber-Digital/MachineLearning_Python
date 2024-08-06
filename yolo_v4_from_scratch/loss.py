import torch
import torch.nn as nn

from utils import intersection_over_union

class YoloLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mse = nn.MSELoss() # needed for box predictions
        self.bce = nn.BCEWithLogitsLoss() # BCE = binary cross-entropy
        # This loss combines a Sigmoid layer and the BCELoss in one single class. 
        # This version is more numerically stable than using a plain Sigmoid followed by a BCELoss as, 
        # by combining the operations into one layer, we take advantage of the log-sum-exp 
        # trick for numerical stability.
        self.entropy = nn.CrossEntropyLoss() # when you have only 1 class - NO multilabel loss
        self.sigmoid = nn.Sigmoid()
        
        
        # constants
        self.lambda_class = 1
        self.lambda_noobj = 10
        self.lambda_obj = 1
        self.lambda_box = 10
        
    def forward(self, predictions, target, anchors): # called thrice for every forward pass (1x per scale)
        obj = target[..., 0] == 1 # check where there is an object in the target
        noobj = target[..., 0] == 0 # if the IOU is greater than 0.5, those should be also ignored
        
        # loss term type "no object found" - simple binary cross entropy loss of base probability
        no_object_loss = self.bce((predictions[...,0:1][noobj], (target[..., 0:1][noobj])))
        
        # loss term type "object detected"
        # need to compute the target 
        # need to reshape the anchors in order to be able to compare with the targets
        anchors = anchors.reshape(1,3,1,1,2) # p_w (=probability coming from network) * exp(t_w)
        
        box_preds = torch.cat([self.sigmoid(predictions[..., 1:3]), torch.exp(predictions[..., 3:5] * anchors)], dim=-1)
        IoU = intersection_over_union(box_preds[obj], target[..., 1:5][obj]).detach() # the IoU should be gradient independent (not changing with gradient)
        # object loss - binary cross entropy loss for in case an object was detected
        object_loss = self.bce((predictions[..., 0:1][obj]), (IoU * target[..., 0:1]))
        
        # loss term type "box coordinate loss"
        predictions[..., 1:3] = self.sigmoid(predictions[..., 1:3]) # mapping x y -> between 0 and 1
        target[..., 3:5] = torch.log(
            (1e-16 + target[..., 3:5] / anchors) # entries 3 to 5 are w, h, class
        ) # this is actually just the inverse of exponential part of the box predictions torch.exp
        # log of 0 is not defined - I got a safety margin in here - 1e-16
        box_loss = self.mse(predictions[..., 1:5][obj], target[..., 1:5][obj])
        
        # loss term type "class correctly detected"
        class_loss = self.entropy(
            (predictions[..., 5:][obj]), (target[..., 5][obj].long()),
        )
        
        return (
            self.lambda_box * box_loss
            + self.lambda_obj * object_loss
            + self.lambda_noobj * no_object_loss
            + self.lambda_class * class_loss
        )