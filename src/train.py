import torch
import torch.nn as nn
import collections
import os
from dataset import provider

if os.path.exists('EndoCV_Catalyst/Input') == False:
  if input("Download Data(Data needs to be stored inside the Input folder which will be done automatically if you press Y)? [Y/n])") == 'Y':
    os.system(f"""mkdir EndoCV_Catalyst/Input""")
    os.system(f"""wget https://ead2020-training-detection-framesonly-phase1.s3.eu-west-2.amazonaws.com/ead2020_semantic_segmentation.zip""")
    os.system(f"""unzip -qq ead2020_semantic_segmentation.zip -d Input""")
from catalyst.contrib.criterion import DiceLoss, IoULoss
from torch import optim
from catalyst.dl import utils
from catalyst.contrib.optimizers import RAdam, Lookahead

# Catalyst has new SOTA optimizers out of box
from models import getModel
num_epochs = int(os.environ['epoch'])
try:
    from catalyst.dl.runner import SupervisedRunner
    from catalyst.dl.callbacks import CheckpointCallback
    from catalyst.contrib.schedulers.onecycle import OneCycleLRWithWarmup
except:
    os.system(f"""pip install -U catalyst""")    
    from catalyst.dl.runner import SupervisedRunner
    from catalyst.dl.callbacks import CheckpointCallback
    from catalyst.contrib.schedulers.onecycle import OneCycleLRWithWarmup
    
from catalyst.dl.callbacks import DiceCallback, IouCallback,  CriterionCallback, CriterionAggregatorCallback

learning_rate = 0.001
encoder_learning_rate = 0.0005

loaders = collections.OrderedDict()
train_loader = provider('train', batch_size=2, num_workers=0,)
valid_loader = provider('val', batch_size=2, num_workers=0,)

model = getModel("FPN", "efficientnet-b3")
logdir = "./logs/segmentation_notebook"
loaders = {
    "train": train_loader,
    "valid": valid_loader
}
layerwise_params = {"encoder*": dict(lr=encoder_learning_rate, weight_decay=0.00003)}
model_params = utils.process_model_params(model, layerwise_params=layerwise_params)
base_optimizer = RAdam(model_params, lr=learning_rate, weight_decay=0.0003)
optimizer = Lookahead(base_optimizer)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.25, patience=2)

# criterion = nn.BCEWithLogitsLoss()
# optimizer = torch.optim.Adam([
#     {'params': model.decoder.parameters(), 'lr': 1e-4}, 
    
#     # decrease lr for encoder in order not to permute 
#     # pre-trained weights with large gradients on training start
#     {'params': model.encoder.parameters(), 'lr': 1e-6},  
# ])
# scheduler = None
runner = SupervisedRunner(device='cuda:0', input_key="image", input_target_key="mask")
# scheduler = OneCycleLRWithWarmup(optimizer, num_steps = 1)
# we have multiple criterions
criterion = {
    "dice": DiceLoss(),
    "iou": IoULoss(),
    "bce": nn.BCEWithLogitsLoss()
}
# runner.train(
#     model=model,
#     criterion=criterion,
#     optimizer=optimizer,
#     loaders=loaders,
#     logdir=logdir,
#     num_epochs=4,
#     scheduler=scheduler,
#     verbose=True
# )
# :

runner.train(
    model=model,
    criterion=criterion,
    optimizer=optimizer,
    scheduler=scheduler,
    
    # our dataloaders
    loaders=loaders,
    
    callbacks=[
        # Each criterion is calculated separately.
        CriterionCallback(
            input_key="mask",
            prefix="loss_dice",
            criterion_key="dice"
        ),
        CriterionCallback(
            input_key="mask",
            prefix="loss_iou",
            criterion_key="iou"
        ),
        CriterionCallback(
            input_key="mask",
            prefix="loss_bce",
            criterion_key="bce"
        ),
        CriterionAggregatorCallback(
            prefix="loss",
            loss_aggregate_fn="weighted_sum", # can be "sum", "weighted_sum" or "mean"
            # because we want weighted sum, we need to add scale for each loss
            loss_keys={"loss_dice": 1.0, "loss_iou": 1.0, "loss_bce": 0.8},
        ),
        DiceCallback(input_key="mask"),
        IouCallback(input_key="mask"),
    ],
    # path to save logs
    logdir=logdir,
    num_epochs=num_epochs,
    main_metric="iou",
    # IoU needs to be maximized.
    minimize_metric=False,
    verbose=True,
)
