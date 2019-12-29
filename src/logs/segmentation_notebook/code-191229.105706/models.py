import os
try:
    import segmentation_models_pytorch as smp
except:
    os.system(f"""pip install -qq git+https://github.com/qubvel/segmentation_models.pytorch""")
    import segmentation_models_pytorch as smp

def getModel(name_, encoder):
    if(name_=='Unet'):
        model = smp.Unet(encoder, encoder_weights='imagenet', classes=5, activation=None)
    elif(name_=='FPN'):
        model = smp.FPN(encoder, encoder_weights='imagenet', classes=5, activation=None)
    elif(name_ == 'Linknet'):
        model = smp.Linknet(encoder, encoder_weights='imagenet', classes=5, activation=None)

    return model    