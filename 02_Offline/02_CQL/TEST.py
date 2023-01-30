import d3rlpy
import gym 
import d4rl

dataset, env = d3rlpy.datasets.get_dataset("hopper-medium-v2")
cql = d3rlpy.algos.CQL(use_gpu = True,)

def eval_policy(policy):
    actions = cql.predict(x)

for t in range(1000000):    
    cql.fit(dataset, n_steps=1, n_steps_per_epoch=1)
    if (t + 1) % args.eval_freq == 0:
        eval_policy(cql)