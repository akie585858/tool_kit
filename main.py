from mylib.utils.trianner import Trainner
from result.ResNet50_dog_breed_argument_plus.setting import *

if __name__ == '__main__':
    # 配置训练器
    trianner = Trainner(
        net=net,
        loss_fn=loss_fn,
        dataset=train_dataset,
        test_dataset=test_dataset,
        counter = train_counter,
        test_counter=test_counter,
        batch_size=batch_size,
        r_batch_size=real_batch,
        saver=saver,
        scheduler = scheduler
    )

    trianner.train()


