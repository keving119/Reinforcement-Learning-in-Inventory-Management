"""@author: KevinG."""
import numpy as np
import random
from inventory_env import InventoryEnv
from cases import General
import scipy.stats as st


case               = General()
case.order_policy  = 'BaseStock'                           # Predetermined order policy, can be either 'X','X+Y' or 'BaseStock'
original_action             = [[0, 0, 0, 185, 0, 0, 40, 0, 0, 40, 0, 0, 40, 0, 0, 40, 0, 40]]
# original_action             = [[37, 47, 33, 63, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30]]
# original_action             = [[82, 100, 64, 83, 35, 35, 35, 35, 35]]
action = (case.action_high - case.action_low) * ((original_action - case.action_min) / (case.action_max - case.action_min)) + case.action_low
replications       = 250

print(action)

for i in range(len(action)):
    totaltotalreward = 0
    totalrewardlist = []
    totalfulfilled = np.zeros([case.no_nodes, case.no_nodes], dtype=int)
    totaldemand = np.zeros([case.no_nodes, case.no_nodes], dtype=int)
    totalbo = np.zeros([case.no_nodes, case.no_nodes], dtype=int)
    for k in range(replications):
        env = InventoryEnv(case, case.action_low, case.action_high, case.action_min,
                        case.action_max, case.state_low, case.state_high)
        run_name = "RN{}".format(k)
        random.seed(k)
        np.random.seed(k)
        totalreward, totalholdingcosts, totalbackordercosts = 0, 0, 0
        _ = env.reset()
        for t in range(case.horizon):
            _, reward, _, info = env.simulate(action[i],False)
            if t >= case.warmup:
                totalholdingcosts += info['holding_costs']
                totalbackordercosts += info['backorder_costs']
                totalreward += reward
        totalrewardlist.append(totalreward)
        totaltotalreward += totalreward
        print(f"bo:{totalbackordercosts}")
        print(totalholdingcosts)
        totalfulfilled += env.TotalFulfilled
        totaldemand += env.TotalDemand
        # totalbo += env.TotalBO
    fillrate = totalfulfilled/totaldemand
    print(fillrate)
    # print(totalrewardlist)
    print(st.t.interval(0.95, len(totalrewardlist)-1, loc=np.mean(totalrewardlist), scale=st.sem(totalrewardlist)))
    print("Average total reward: {}, Action: {}".format(totaltotalreward/replications, action[i]))