from torch import Tensor, no_grad, save
from torch.cpu import device_count
from torch.nn import Module
from torch.utils.data import Dataset, IterableDataset, DataLoader
from torch.optim.lr_scheduler import LinearLR
from mylib.utils.recorder import Counter, Saver
from tqdm import trange
import pynvml
import os

pynvml.nvmlInit()

class Trainner():
    def __init__(self, net:Module, loss_fn:Module, dataset:Dataset|IterableDataset, test_dataset:Dataset|IterableDataset,  counter:Counter, test_counter:Counter, saver:Saver, batch_size:int, r_batch_size:int, device:str='cuda') -> None:
        self.device = device

        # 模型配置
        self.net = net.to(device)
        self.loss_fn = loss_fn.to(device)
        self.data_set = dataset
    
        self.counter = counter
        self.test_counter = test_counter
        self.saver = saver

        # 处理输入输出中间值

        # batch_size设置
        self.batch_size = batch_size
        self.get_batchsize_userate(r_batch_size)
        self.fold_num = self.batch_size // batch_size
    
        # 初始化dataloader
        num_workers = min([8, device_count(), batch_size])
        if dataset.__class__.__bases__[0]==Dataset:
            shuffle = True
        else:
            shuffle = False
        self.data_loader = DataLoader(dataset, r_batch_size, shuffle, drop_last=True, num_workers=num_workers)
        self.test_loader = DataLoader(test_dataset, r_batch_size, shuffle, drop_last=True, num_workers=num_workers)

        # 计算一个epoch中的iter_nums
        data_length = len(dataset)
        test_data_length = len(test_dataset)
        if dataset.__class__.__bases__[0] == IterableDataset:
            self.iter_nums = (data_length // num_workers) // self.batch_size * num_workers + (data_length % num_workers) // self.batch_size
            self.test_iter_nums = (test_data_length // num_workers) // self.batch_size * num_workers + (test_data_length % num_workers) // self.batch_size
        else:
            self.iter_nums = data_length // self.batch_size
            self.test_iter_nums = test_data_length // self.batch_size

    def train(self, schedul_list:list, test_rate:int):
        '''
        schedul_list:[{
            optim:优化器，
            lr_scheduler:变换策略
            epochs_num:控制epoch数
            describe:策略描述
        }]
        '''

        describtion = ''
        count = 0

        train_result = []
        test_result = []

        for schedul in schedul_list:
            self.optim = schedul['optim']
            self.scheduler = schedul['lr_scheduler']

            describtion += schedul['describe'] + ':'
            for epoch in range(schedul['epochs_num']):
                if isinstance(schedul['lr_scheduler'], LinearLR):
                    mode = 'iter'
                elif schedul['lr_scheduler'] is None:
                    mode = None
                else:
                    mode = 'epoch'
                self.train_one_epoch(mode, describtion=describtion+f'[{epoch+1}/{schedul['epochs_num']}]')
                train_result.append(self.counter.close())

                count += 1
                if count % test_rate == 0:
                    self.train_one_epoch(is_trian=False, describtion='test epoch')
                    test_result.append(self.test_counter.close())

                    self.saver.save(self.net, train_result, test_result)
                    
                    train_result = []
                    test_result = []

            describtion += str(schedul['epochs_num']) + ' '
    
    def train_one_epoch(self, mode='epoch', is_trian=True, describtion:str=None):
        if is_trian:
            data_loader = self.data_loader
            iter_nums = self.iter_nums
        else:
            data_loader = self.test_loader
            iter_nums = self.test_iter_nums
       
        with trange(iter_nums) as tbar:
            if not(describtion is None):
                tbar.set_description(describtion + f' lr: {self.optim.param_groups[0]['lr']:.3f}')
            for index, (x, y) in enumerate(data_loader):
                if is_trian:
                    self.net.train()
                    if index % self.fold_num == 0:
                        self.optim.zero_grad()
                    loss, y_hat = self.train_one_iter(x, y)
                    loss.backward()

                    if (index+1) % self.fold_num == 0:
                        self.optim.step()
                        if mode == 'iter':
                            self.scheduler.step()
                            tbar.set_description(describtion + f' lr: {self.scheduler.get_last_lr()[0]}')
                    result = self.counter.update(y=y, y_hat=y_hat, loss=loss)

                else:
                    self.net.eval()
                    with no_grad():
                        loss, y_hat = self.train_one_iter(x, y)
                    result = self.test_counter.update(y=y, y_hat=y_hat, loss=loss)

                if (index+1) % self.fold_num == 0:
                    tbar.set_postfix(**result)
                    tbar.update()
        if is_trian & (mode=='epoch'):
            self.scheduler.step()   


    def train_one_iter(self, x:Tensor, y:Tensor) -> tuple[Tensor, Tensor] | Tensor:
        x = x.to(self.device)
        y = y.to(self.device)

        y_hat = self.net(x)
        loss = self.loss_fn(y_hat, y)

        return loss, y_hat

    # 计算gpu利用率
    def get_batchsize_userate(self, batch_size):
        pynvml.nvmlInit()

        def get_gpu_use_rate():
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            memoryInfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
            total_memory = memoryInfo.total
            used_memory = memoryInfo.used
            use_rate = used_memory / total_memory
            return use_rate

        def get_batch_use_rate(batch_size):
            in_p = DataLoader(self.data_set, batch_size, drop_last=True)
            count = 0
            max_use_rate = 0
            for x, _ in in_p:
                x = x.to(self.device)
                self.net(x)
                use_rate = get_gpu_use_rate()
                if use_rate > max_use_rate:
                    max_use_rate = use_rate
                count+= 1
                if count > 10:
                    break

            return max_use_rate

        pre_batch_use = get_batch_use_rate(batch_size)

        print(f'batch_size: {batch_size}  use_rate: {pre_batch_use}')


if __name__ == '__main__':
    pass
