import inspect
from torch import no_grad
import torch
import os

class Recorder():
    def __init__(self, **func_set) -> None:
        '''
        记录类对象基础类
        '''
        self.funcs = func_set
        # 初始化指标
        self.init()
    
    @no_grad()
    def update(self, **in_p) -> dict:
        for attr in self.funcs:
            pre_func = self.funcs[attr][0]
            attr_list = getattr(self, attr+'_list')

            params = inspect.signature(pre_func).parameters
            p = []
            for param in params:
                p.append(in_p[param])
            result = pre_func(*p)
            attr_list.append(result)
    
    def init(self):
        for attr in self.funcs:
            setattr(self, attr, 0)
            setattr(self, attr+'_list', [])

    def close(self):
        result_set = {}
        for attr in self.funcs:
            attr_list = getattr(self, attr+'_list')
            attr_list = list(zip(*attr_list))

            attr_func = self.funcs[attr][1]
            attr_result = attr_func(*attr_list)
            result_set[attr] = attr_result

        self.init()

        return result_set


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
            
            current_result = ((self.count-1)/self.count) * last_result + result / self.count
            result_set[attr] = current_result
            setattr(self, attr, current_result)

        return result_set
    
    def init(self):
        for attr in self.funcs:
            setattr(self, attr, 0)

        self.count = 0.0

    def close(self):
        result = {}
        for attr in self.funcs:
            result[attr] = getattr(self, attr)

        self.init()

        return result


class Saver():
    def __init__(self, save_root, eval_attr, test_rate, epochs_num=0, best_val=None):
        self.save_root = save_root
        self.best_val = best_val
        self.eval_attr = eval_attr
        self.test_rate = test_rate
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

if __name__ == '__main__':
    from itertools import chain
    a = [(1,2, 3), (4, 5, 6), (7, 8, 9), (1,)]
    b = list(chain(*a))
    print(b)

    
