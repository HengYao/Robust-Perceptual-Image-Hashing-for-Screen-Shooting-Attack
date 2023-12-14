import torch
import torch.nn as nn
# from model.encoder_sw2 import Deephash
from model.encoder_SwinUnet import Deephash
from option.options import HiDDenConfiguration

class Hidden:
    def __init__(self, configuration: HiDDenConfiguration, device: torch.device):

        super(Hidden, self).__init__()
        self.deephash = Deephash(configuration).to(device)
        self.optimizer = torch.optim.Adam(self.deephash.parameters(),lr=0.0001)
        self.config = configuration
        self.device = device
        self.mse_loss = nn.MSELoss().to(device)


    def train_on_batch(self, images):

        batch_size = images.shape[0]

        with torch.enable_grad():

            before, after, hash_code = self.deephash(images)

            # before, hash_code = self.deephash(images)

            loss_sim_sum = 0
            loss_dif_sum = 0
            loss_dif_avg_sum = 0

            self.optimizer.zero_grad()

            for num_in_fea in range(1, 46):

                loss_sim = self.mse_loss(hash_code[0], hash_code[num_in_fea])

                loss_sim = torch.sigmoid(loss_sim)


                loss_sim_sum = loss_sim_sum + loss_sim

            for num_in_fea1 in range(46,91):
                loss_dif = self.mse_loss(hash_code[0], hash_code[num_in_fea1])
                loss_dif_avg = self.mse_loss(after[0],after[num_in_fea1])

                loss_dif = torch.sigmoid(loss_dif)
                loss_dif_avg = torch.sigmoid(loss_dif_avg)


                loss_dif_sum = loss_dif_sum + loss_dif
                loss_dif_avg_sum = loss_dif_avg_sum + loss_dif_avg

            loss = (loss_sim_sum/45) - (loss_dif_sum/45) - 0.05*(loss_dif_avg_sum/45)

            loss.backward(retain_graph=True)

            self.optimizer.step()

        losses = {
            'similar_loss    ': (loss_sim_sum/45).item(),
            'different_loss  ': (loss_dif_sum/45).item(),
            'avgpool_loss_ba ': (loss_dif_avg_sum/45).item(),
            'loss            ': loss.item()
        }

        return losses, hash_code


    def validate_on_batch(self, images):

        batch_size = images.shape[0]
        loss = []
        with torch.no_grad():

            _, _, hash_code = self.deephash(images)

            for num_in_fea in range(1, batch_size):
                loss.append(self.mse_loss(hash_code[0], hash_code[num_in_fea]))

        return loss, hash_code

    def test_single(self,images):

        with torch.no_grad():

            before_a,after_a,hash_code = self.deephash(images)

        return before_a,after_a,hash_code

    def to_stirng(self):
        return '{}\n'.format(str(self.deephash))