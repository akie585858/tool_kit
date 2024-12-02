from torch import Tensor, no_grad
from torch.cpu import device_count
from torch.nn import Module
from torch.utils.data import Dataset, IterableDataset, DataLoader
from mylib.utils.recorder import Counter, Saver, Tbar
import pynvml
import re

pynvml.nvmlInit()

# 控制训练及展示，中断处理
class Scheduler():
    def __init__(self, complete_nums, test_rate):
        self.schedulers = []
        self.count = 0
        self.test_rate = test_rate

        self.complete_nums = complete_nums

    # 初始配置，添加训练策略
    def append(self, name, mode:str, optim, epoch_nums, lr_scheduler):
        self.schedulers.append({
            'name':name,
            'mode':mode,
            'optim':optim,
            'epoch_nums':epoch_nums,
            'lr_scheduler':lr_scheduler
        })
    
    # 完成配置，进行策略统计与中断处理
    def close(self):
        # 中断处理
        Tbar.set_scheduler(self.schedulers, self.complete_nums)

        process_result = []
        count = self.complete_nums
        for scheduler in self.schedulers:
            epoch_nums = scheduler['epoch_nums']
            if count - epoch_nums > 0:
                count -= epoch_nums
            else:
                scheduler['epoch_nums'] -= count
                process_result.append(scheduler)
                count = 0
            
        return self.schedulers

class Trainner():
    def __init__(self, net:Module, loss_fn:Module, dataset:Dataset|IterableDataset, test_dataset:Dataset|IterableDataset,  counter:Counter, test_counter:Counter, saver:Saver, scheduler:Scheduler, batch_size:int, r_batch_size:int, device:str='cuda') -> None:
        self.device = device

        # 模型配置
        self.net = net.to(device)
        self.loss_fn = loss_fn.to(device)
        self.data_set = dataset
    
        self.counter = counter
        self.test_counter = test_counter
        self.saver = saver
        self.scheduler = scheduler
        self.saver.set_test_rate(self.scheduler.test_rate)

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

    # 根据训练策略选择训练函数开始训练
    def train(self):
        schedul_list = self.scheduler.close()
        test_rate = self.scheduler.test_rate

        describtion = ''
        count = 0

        train_result = []
        test_result = []

        for schedul in schedul_list:
            self.optim = schedul['optim']
            self.scheduler = schedul['lr_scheduler']

            for _ in range(schedul['epoch_nums']):
                mode = schedul['mode']
                if (mode is None) or (self.scheduler is None):
                    mode = 'epoch'
                
                if mode == 'epoch':
                    result = self.train_epoch_byepoch()
                elif mode == 'iter':
                    result = self.train_epoch_byiter()

                train_result.append(result)

                count += 1
                if count % test_rate == 0:
                    result = self.test_one_epoch()
                    test_result.append(result)
                    self.saver.save(self.net, train_result, test_result)
                    
                    train_result = []
                    test_result = []

            describtion += str(schedul['epoch_nums']) + ' '

# epoch模式选择----------------------------------------------------
    # 训练一个epoch，并按epoch迭代
    def train_epoch_byepoch(self):
        # 获取数据加载器与迭代数
        data_loader = self.data_loader

        self.net.train()
        tbar = Tbar(data_loader, True)
        lr = round(self.optim.param_groups[0]['lr'], 8)
        tbar.set_description(tbar.description + f' lr: {lr}')
    
        for index, (x, y) in enumerate(tbar):
            if index % self.fold_num == 0:
                self.optim.zero_grad()
            loss, y_hat = self.train_one_iter(x, y)
            loss.backward()

            result = self.counter.update(y=y, y_hat=y_hat, loss=loss)
            if (index+1) % self.fold_num == 0:
                self.optim.step()
                tbar.set_postfix(**result)
    
        if not self.scheduler is None:
            self.scheduler.step()   

        epoch_result = self.counter.close()
        # tbar.set_postfix(**epoch_result)
        # tbar.update(0)

        return epoch_result

    # 训练一个epoch，并按iter迭代
    def train_epoch_byiter(self):
        # 获取数据加载器与迭代数
        data_loader = self.data_loader

        self.net.train()
        tbar = Tbar(data_loader, True)
        lr = round(self.optim.param_groups[0]['lr'], 8)
        tbar.set_description(tbar.description + f' lr: {lr}')
       
        for index, (x, y) in enumerate(tbar):
            self.net.train()
            if index % self.fold_num == 0:
                self.optim.zero_grad()
            loss, y_hat = self.train_one_iter(x, y)
            loss.backward()

            result = self.counter.update(y=y, y_hat=y_hat, loss=loss)
            if (index+1) % self.fold_num == 0:
                self.optim.step()
                self.scheduler.step()

                # 更新lr描述
                lr = round(self.optim.param_groups[0]['lr'], 8)
                tbar.set_description(tbar.description + f' lr: {lr}')
                tbar.set_postfix(**result)
        
        epoch_result = self.counter.close()
        # tbar.set_postfix(**epoch_result)
        # tbar.update(0)

        return epoch_result

    # 测试一个epoch   
    def test_one_epoch(self):
        data_loader = self.test_loader
       
        tbar = Tbar(data_loader, False)
        tbar.set_description('test')
        self.net.eval()
        for index, (x, y) in enumerate(tbar):
            with no_grad():
                loss, y_hat = self.train_one_iter(x, y)
            result = self.test_counter.update(y=y, y_hat=y_hat, loss=loss)

            if (index+1) % self.fold_num == 0:
                tbar.set_postfix(**result)
        
        epoch_result = self.test_counter.close()
        # tbar.set_postfix(**epoch_result)
        # tbar.update(0)

        return epoch_result

# ----------------------------------------------------------------- 
    # 训练一代
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

# 未完成
class FlowChain():
    def __init__(self, *links, **components):
        # 获取流动链中所有组件
        all_component = []
        for link in links:
            comp = re.split('->|,', link)
            comp = [c.strip() for c in comp]
            all_component += comp
        all_component = list(set(all_component))

        # 创建中间变量
        for c in all_component:
            if not c in components:
                setattr(self, c, None)

if __name__ == '__main__':
    pass
