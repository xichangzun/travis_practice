from test_mobileNetV2 import save_filename
import torch
from Models.mobileNetV2 import mobileNet_V2

the_model=mobileNet_V2(pretrained=True)
the_model.load_state_dict(torch.load(save_filename))
print (the_model)

