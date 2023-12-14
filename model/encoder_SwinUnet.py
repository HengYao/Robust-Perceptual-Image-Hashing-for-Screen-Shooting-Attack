from option.options import HiDDenConfiguration
from model.SwinUnet import *
class Deephash(nn.Module):

    def __init__(self,config: HiDDenConfiguration):

        super(Deephash, self).__init__()
        self.hidden_layers = SwinUnet()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(96, 96)
        self.activation1 = nn.ReLU()
        self.fc2 = nn.Linear(96, 96)
        self.activation2 = nn.ReLU()
        self.fc3 = nn.Linear(96, config.L)
        self.hash_layer = nn.Sequential(self.fc1, self.activation1, self.fc2, self.activation2, self.fc3)



    def forward(self, image):

        out_before_avg = self.hidden_layers(image)
        out_after_avg = self.avgpool(out_before_avg)
        out = out_after_avg.view(out_after_avg.size(0), -1)
        hash = self.hash_layer(out)

        return out_before_avg, out_after_avg, hash


if __name__ == '__main__':
    a = torch.rand(1,3,224,224)
    hidden_config = HiDDenConfiguration(H=224, W=224, L=50)
    model = Deephash(hidden_config)
    _,_,b = model(a)
    print(b)