import torch.cuda
import wandb
import math
from models.save_utils import * #needs to be added

from tqdm import tqdm


class LrScheduler():
    """ Implements a learning rate schedule with warum up and decay """

    def __init__(self, peak_lr=4e-4, peak_it=10000, decay_rate=0.5, decay_it=100000):
        self.peak_lr = peak_lr
        self.peak_it = peak_it
        self.decay_rate = decay_rate
        self.decay_it = decay_it

    def get_cur_lr(self, it):
        if it < self.peak_it:  # Warmup period
            return self.peak_lr * (it / self.peak_it)
        it_since_peak = it - self.peak_it
        return self.peak_lr * (self.decay_rate ** (it_since_peak / self.decay_it))


class Trainer:
    def __init__(self,dl,model,configs):
        self.dl = dl
        self.model = model
        self.configs = configs
        # lr_scheduler = LrScheduler(peak_lr=1e-4, peak_it=2500, decay_it=4000000, decay_rate=0.16)
        # self.optim = torch.optim.Adam(self.model.parameters(),lr = lr_scheduler.get_cur_lr(0))
        # Currently something is wrong with the lr_scheduler, Just use lr as 0.0001. This will be modified later.
        self.optim = torch.optim.Adam(self.model.parameters(),lr=0.0001)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def run(self):
        record = math.inf
        for i in tqdm(range(self.configs['Epoch'])):
            self.model.train()
            for b in self.dl['train']:
                self.optim.zero_grad()
                image = b[0].unsqueeze(0).to(self.device)
                true_pose = b[1].unsqueeze(0).to(self.device)
                estim_pose = self.model(image,image)

                error = ((true_pose - estim_pose)**2).mean()
                error.backward()

                self.optim.step()
                print('epoch: {}, loss: {}'.format(str(i),str(error)))
                wandb.log({'epoch':i,'L2-Loss':error})

            # self.model.eval()
            # with torch.no_grad():
            #     for b in self.dl['test']:
            #         image = b[0].to(self.device)
            #         true_pose = b[1].to(self.device)
            #         estim_pose = self.model(image, image)
            #
            #         error = ((true_pose - estim_pose) ** 2).mean()
            #
            #         if error <= record:
            #             record = error
            #             print('New Best Score : {} at epoch {}'.format(str(error),str(i)))
            #     wandb.log({'Test-Data Score' : error})