import inspect
from torch import no_grad
import torch
import os
from tqdm import tqdm

class Counter():
    def __init__(self, **func_set) -> None:
        '''
        计数对象基础类，用于可迭代更新的评价指标。
        update函数以字典的方式接收结果变量
        对象会自动执行指标更新，只需自定义结果到最终指标的
        计算函数即可
        '''

        self.funcs = func_set
        # 初始化指标
        self.init()
    
    @no_grad()
    def update(self, **in_p) -> dict:
        self.count += 1

        result_set = {}
        for attr in self.funcs:
            func = self.funcs[attr]
            last_result = getattr(self, attr)

            params = inspect.signature(func).parameters
            p = []
            for param in params:
                p.append(in_p[param])
            result = func(*p)

            if not result is None:
                current_result = ((self.count-1)/self.count) * last_result + result / self.count
                result_set[attr] = current_result
                setattr(self, attr, current_result)
            else:
                setattr(self, attr, None)

        return result_set
    
    def init(self):
        for attr in self.funcs:
            setattr(self, attr, 0)

        self.count = 0.0

    def close(self):
        results = {}
        for attr in self.funcs:
            result = getattr(self, attr)
            if result is None:
                results[attr] = self.funcs[attr].close()
            else:
                results[attr] = result

        self.init()

        return results


class Saver():
    def __init__(self, save_root, eval_attr, epochs_num=0, best_val=None):
        self.save_root = save_root
        self.best_val = best_val
        self.eval_attr = eval_attr
        self.test_rate = 1
        self.epochs_num = epochs_num

        if os.path.exists(os.path.join(self.save_root, 'train_result.pt')):
            self.train_result = torch.load(os.path.join(self.save_root, 'train_result.pt'))
            self.test_result = torch.load(os.path.join(self.save_root, 'test_result.pt'))
        else:
            self.train_result = []
            self.test_result = []

        os.makedirs(self.save_root, exist_ok=True)

    def save(self, net, train_result, test_result):
        self.train_result += train_result
        self.test_result += test_result

        torch.save(net.state_dict(), os.path.join(self.save_root, 'last.pt'))
        torch.save(self.train_result, os.path.join(self.save_root, 'train_result.pt'))
        torch.save(self.test_result, os.path.join(self.save_root, 'test_result.pt'))

        eval_val = test_result[-1][self.eval_attr]
        if self.best_val is None:
            self.best_val = eval_val
            torch.save(net.state_dict(), os.path.join(self.save_root, 'best.pt'))
        else:
            if eval_val > self.best_val:
                self.best_val = eval_val
                torch.save(net.state_dict(), os.path.join(self.save_root, 'best.pt'))

        self.epochs_num += self.test_rate
        with open(os.path.join(self.save_root, 'last.tmp'), 'w') as f:
            f.write(f'{self.epochs_num}\n{self.best_val}')

    def set_test_rate(self, test_rate):
        self.test_rate = test_rate


class Tbar(tqdm):
    count = 0
    scheduler = None
    def __init__(self, iter_item, is_train):
        super().__init__(iter_item)
        if is_train:
            Tbar.count += 1
        desciption = ''
        count = Tbar.count


        # 设置tbar描述
        for name, epoch_nums in Tbar.scheduler:
            if count - epoch_nums > 0:
                desciption += f'{name}:{epoch_nums} '
                count -= epoch_nums
            else:
                desciption += f'{name}:[{count}/{epoch_nums}]'
                break
        self.description = desciption


    @staticmethod
    def set_scheduler(schedulers, complete_num):
        '''设置tbar_scheduler'''
        epoch_list = []
        for scheduler in schedulers:
            epoch_list.append((scheduler['name'], scheduler['epoch_nums']))
        Tbar.scheduler = epoch_list
        Tbar.count = complete_num

if __name__ == '__main__':
    pass

    
