import json
import os
from torch.optim.lr_scheduler import MultiStepLR, LinearLR

def process_setting(json_file):
    with open(json_file, 'r') as f:
        json_result = json.load(f)

    train_set = json_result['train']

    # 参数配置
    save_root = train_set['save_root']
    eval_attr = train_set['eval_attr']
    pretrain_file = train_set['pretrain_file']
    batch_size = train_set['batch_size']
    real_batch = train_set['real_batch']

    if os.path.exists(os.path.join(save_root, 'last.tmp')):
        with open(os.path.join(save_root, 'last.tmp'), 'r') as f:
            complete_nums = int(f.readline())
            best_val = float(f.readline())
    else:
        complete_nums=0
        best_val = None

    return complete_nums, best_val, pretrain_file, save_root, eval_attr, batch_size, real_batch


class MyMultiStepLR(MultiStepLR):
    def __init__(self, complete_nums, optimizer , milestones:list, gamma ):
        result_stones = []
        start = False
        for stone in milestones:
            if complete_nums-stone < 0:
                start = True
            if start:
                if complete_nums < 0:
                    complete_nums = 0
                result_stones.append(stone-complete_nums)

        super().__init__(optimizer=optimizer, milestones=result_stones, gamma=gamma)

class MyLinearLR(LinearLR):
    def __init__(self, complete_nums, optimizer, start_factor, end_factor, total_iters):
        # if complete_nums < 0:
        super().__init__(optimizer, start_factor, end_factor, total_iters)
