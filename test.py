import torch
from torch import nn

class MyNet(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.head = nn.Linear(4, 10)
        self.classfiy = nn.Linear(10, 5)

    def forward(self, x):
        x = self.head(x)
        return self.classfiy(x)
    
if __name__ == '__main__':
    params = torch.load('test.pt')
    attr_name = 'head'
    param_set = {}
    for param in params:
        attr = param.split('.')
        if attr[0] == attr_name:
            param_set[attr[1]] = params[param]


    net = MyNet()
    net.head.load_state_dict(param_set)
