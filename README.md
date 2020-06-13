# Discriminant Distribution-Agnostic loss (DDA loss)
This is the PyTorch implementation of our CVPR 2020 Workshop paper: [**Discriminant Distribution-Agnostic Loss for Facial Expression Recognition in the Wild**](http://openaccess.thecvf.com/content_CVPRW_2020/papers/w29/Farzaneh_Discriminant_Distribution-Agnostic_Loss_for_Facial_Expression_Recognition_in_the_Wild_CVPRW_2020_paper.pdf).

Workshop: [Challenges and Promises of Inferring Emotion from Images and Video](http://cbcsl.ece.ohio-state.edu/cvpr-2020/index.html)

**Update 07/12/2020:** The loss function is uploaded. The full training code and the models will be available soon.

### Requirements
```bash
python => 3.7
pytorch => 1.4
```

### Usage
```python
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from models import resnet18

from loss import DDALoss

if torch.cuda.is_available():
    device = torch.device('cuda')
    cudnn.benchmark = True
else:
    device = torch.device('cpu')

criterion = {
    'softmax': nn.CrossEntropyLoss().to(device),
    'dda': DDALoss(num_classes=7, feat_size, lamb=0.01, gamma=4.0).to(device)
    }
optimizer = {
    'softmax': torch.optim.SGD(model.parameters(), 0.01,
                               momentum=0.9,
                               weight_decay=0.0005),
    'dda': torch.optim.SGD(criterion['dda'].parameters(), 0.5)
    }

# in training loop:
feat, output = model(images)
l_softmax = criterion['softmax'](output, target)
l_added, l_center, l_dda = criterion['dda'](feat, target)
l_total = l_softmax + l_added
```
**NOTE:** The full training code will be available soon
