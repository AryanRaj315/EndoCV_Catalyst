import torch
import torch.nn as nn
import collections
import os
from dataset import provider
# os.system(f"""ls""")
from models import getModel

try:
    from catalyst.dl.runner import SupervisedRunner
    from catalyst.dl.callbacks import CheckpointCallback
    from catalyst.contrib.schedulers.onecycle import OneCycleLRWithWarmup
except:
    os.system(f"""pip install -U catalyst""")    
    from catalyst.dl.runner import SupervisedRunner
    from catalyst.dl.callbacks import CheckpointCallback
    from catalyst.contrib.schedulers.onecycle import OneCycleLRWithWarmup

    
loaders = collections.OrderedDict()
train_loader = provider('train', batch_size=2, num_workers=0,)
valid_loader = provider('val', batch_size=2, num_workers=0,)

model = getModel("FPN", "efficientnet-b3")
logdir = "./logs/segmentation_notebook"
loaders = {
    "train": train_loader,
    "valid": valid_loader
}
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam([
    {'params': model.decoder.parameters(), 'lr': 1e-4}, 
    
    # decrease lr for encoder in order not to permute 
    # pre-trained weights with large gradients on training start
    {'params': model.encoder.parameters(), 'lr': 1e-6},  
])
scheduler = None
runner = SupervisedRunner()
scheduler = OneCycleLRWithWarmup(optimizer, num_steps = 1)

runner.train(
    model=model,
    criterion=criterion,
    optimizer=optimizer,
    loaders=loaders,
    logdir=logdir,
    num_epochs=4,
    scheduler=scheduler,
    verbose=True
)