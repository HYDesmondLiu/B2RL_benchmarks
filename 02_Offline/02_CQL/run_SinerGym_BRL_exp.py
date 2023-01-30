import argparse
import os
import random
from random import randrange

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_rand", default=False)
    args = parser.parse_args()

    # Define params lists

    avail_gpu_list = [0,1,2,3,4,5,6]
    envs = [
        'Eplus-5Zone-hot-continuous-stochastic-v1',
        'Eplus-5Zone-cool-continuous-stochastic-v1',
        'Eplus-5Zone-mixed-continuous-stochastic-v1',

        'Eplus-5Zone-hot-continuous-v1',
        'Eplus-5Zone-mixed-continuous-v1',
        'Eplus-5Zone-cool-continuous-v1',
        ]
    buffer_folders = {
        '/data/hsinyu/01_Building/sinergym/01_DDPG/buffers/': 'Expert_DDPG',
        #'/data/hsinyu/01_Building/sinergym/04_RBC/buffers/': 'RBC',
        #'/data/hsinyu/01_Building/sinergym/00_Random/buffers/': 'Random',
    }

    seed_list = [0,1,2] #[0,1,2,3,4] 
    policy = ['CQL']

    for seed in seed_list:
        for e in envs:
            for p in policy:
                for folder, buffer_name in buffer_folders.items():
                    gpu_id = randrange(8) if args.gpu_rand else random.choice(avail_gpu_list)

                    command = f'\nCUDA_VISIBLE_DEVICES={gpu_id} python -u Sinergym_BRL.py --seed {seed} \
                        --env {e} --policy {p} --buffer_folder {folder} | tee {e}_{seed}_{p}_{buffer_name}.log &'
                    print(command)        
                    os.system(command)
