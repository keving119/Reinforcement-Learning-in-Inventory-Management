# Resulting policy figures

from inventory_env import InventoryEnv
# from EnvironmentBenchmarks import BenchmarkCalculations
from ppo.ppo_buffer_functions import Buffer
from ppo.ppo_support_functions import scale_input, set_seeds
from ppo.ppo_network_functions import MLPActorCritic
from cases import BeerGame, Divergent, General
from gym.spaces import Box, Discrete
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import seaborn as sns
import scipy.stats
import random
import itertools
import math
from scipy import ndimage

CASE = "CBC"
FIX = True


def myround(x, base=5):
    return base * math.ceil(x / base)


def check_action_space(action):
    low = case.action_low
    high = case.action_high
    max = case.action_max
    min = case.action_min
    action_clip = np.clip(action, low, high)
    for i in range(len(action_clip)):
        action_clip[i] = ((action_clip[i] - low[i]) / (high[i] - low[i])) * ((max[i] - min[i])) + min[i]
    action = [np.round(num) for num in action_clip]
    return action

if CASE == "Divergent":
    case = Divergent()
    case2 = Divergent()
    case2.order_policy = "BaseStock"
    basestock = [124, 30, 30, 30]
    begin_sim = 25
    end_sim = 75
    directory = 'results/DRL/' + CASE + '/'
elif CASE == "CBC":
    case = General()
    case2 = General()
    case2.order_policy = "BaseStock"
    basestock = [37, 47, 33, 63, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30]
    basestock_action = (case.action_high - case.action_low) * ((basestock - case.action_min) / (case.action_max - case.action_min)) + case.action_low
    begin_sim = 50
    end_sim = 100
    directory = 'result_files/CBC/CAPtestechtv2/'

env = InventoryEnv(case, case.action_low, case.action_high, case.action_min, case.action_max,
                   case.state_low, case.state_high)

# fill buffer with final network
cooldown_buffer = False

def save_states_encountered(env, rn, size, model):
    ac = MLPActorCritic(env.observation_space, env.action_space, env.feasible_actions, (64, 64), nn.Tanh, 0.0,
                        'uniform')
    saved_state = torch.load(directory + model)
    ac.load_state_dict(saved_state['structure'])
    set_seeds(env, rn)
    print(rn)
    # instantiate buffer and logger, tensorboard writer
    if isinstance(env.action_space, Box):
        action_dim = env.action_space.shape
        action_dim_ent = action_dim[0]
    else:
        raise NotImplementedError

    buffer = Buffer(env.observation_space.shape[0], action_dim,
                    size, 0.99, 0.95, cooldown=cooldown_buffer)
    buffer.reset_buffer()
    o = env.reset()
    # fill up buffer
    for t in range(buffer.max_size):
        o = torch.as_tensor([scale_input(env, o)], dtype=torch.float32)
        a, v, logp, entropy, stdev = ac.step(o, True)
        next_o, r, d, info_dict = env.simulate(a[0])
        # save and log observation, action, reward (scaled by number of iterations), value, logprob, entropy
        buffer.store(o, a, r / size, v, logp, entropy)
        # update obs (very important!)
        o = next_o
        # determine if the buffer has filled up
        buffer_finished = t == buffer.max_size - 1

        # we need to get the final value estimate
        if buffer_finished:
            _, v, _, _, _ = ac.step(torch.as_tensor([scale_input(env, o)], dtype=torch.float32))
            buffer.finish_path(v)
            totalfulfilled = env.TotalFulfilled
            totaldemand = env.TotalDemand
            fillrate = totalfulfilled / totaldemand
            print("DRL")
            print(fillrate)
            o = env.reset()
    buffer.get()
    state_norm = buffer.total_data['obs'].numpy()
    state = ((state_norm - 0) / (1 + 0)) * ((env.state_high - env.state_low)) + env.state_low
    state = np.rint(state)
    
    # write the states to an excel so we can check what states we encounter
    state_excel = pd.DataFrame(state)
    state_excel.to_excel('states_{}.xlsx'.format(rn))

def get_ppo_policy_results(env, rn, size, df, model):
    ac = MLPActorCritic(env.observation_space, env.action_space, env.feasible_actions, (64, 64), nn.Tanh, 0.0,
                        'uniform')
    saved_state = torch.load(directory + model)
    print(saved_state.keys())
    ac.load_state_dict(saved_state['structure'])
    set_seeds(env, rn)
    print(rn)
    # instantiate buffer and logger, tensorboard writer
    if isinstance(env.action_space, Box):
        action_dim = env.action_space.shape
        action_dim_ent = action_dim[0]
    else:
        raise NotImplementedError

    buffer = Buffer(env.observation_space.shape[0], action_dim,
                    size, 0.99, 0.95, cooldown=cooldown_buffer)
    buffer.reset_buffer()
    o = env.reset()
    holdinglist, bolist, rewardlist = [], [], []
    # fill up buffer
    for t in range(buffer.max_size):
        o = torch.as_tensor([scale_input(env, o)], dtype=torch.float32)
        a, v, logp, entropy, stdev = ac.step(o, True)
        next_o, r, d, info_dict = env.simulate(a[0])
        # save and log observation, action, reward (scaled by number of iterations), value, logprob, entropy
        buffer.store(o, a, r / size, v, logp, entropy)
        # update obs (very important!)
        o = next_o
        # determine if the buffer has filled up
        buffer_finished = t == buffer.max_size - 1
        if t >= begin_sim and t < end_sim:
            holdinglist.append(info_dict['holding_costs'])
            bolist.append(info_dict['backorder_costs'])
            rewardlist.append(r)
            df = df._append({'RN': rn,
                            'PPO_Inventory': info_dict['holding_costs'],
                            'PPO_Backorders': info_dict['backorder_costs'],
                            'PPO_Costs': r}, ignore_index=True)

        # we need to get the final value estimate
        if buffer_finished:
            _, v, _, _, _ = ac.step(torch.as_tensor([scale_input(env, o)], dtype=torch.float32))
            buffer.finish_path(v)
            totalfulfilled = env.TotalFulfilled
            totaldemand = env.TotalDemand
            fillrate = totalfulfilled / totaldemand
            print("DRL")
            print(fillrate)
            o = env.reset()
    buffer.get()
    state_norm = buffer.total_data['obs'].numpy()
    state = ((state_norm - 0) / (1 + 0)) * ((env.state_high - env.state_low)) + env.state_low
    state = np.rint(state)
    inventory = state[:, 0]
    if env.case.__class__.__name__ == "BeerGame":
        backorders = state[:, 6:10]
    elif env.case.__class__.__name__ == "Divergent":
        backorders = state[:, 6:9]
    elif env.case.__class__.__name__ == "General":
        backorders = state[:, 1]
    else:
        raise NotImplemented

    
    lengthx = 101
    x_axis = np.arange(-(lengthx/2), lengthx/2, dtype=int)
    # Inventory upstream
    lengthy = 120 # 66
    y_axis = np.arange(lengthy)
    #in Transit
    lengthy2 = 76
    y_axis2 = np.arange(lengthy2)
    data, data2, data3, data4, data5, data6 = np.zeros([lengthy, lengthx]), np.zeros([lengthy, lengthx]), np.zeros([lengthy, lengthx]), np.zeros([lengthy2, lengthx]), np.zeros([lengthy2, lengthx]), np.zeros([lengthy2, lengthx])
    # Warehouse
    x_axis_warehouse = np.arange(lengthx)
    y_axis_warehouse = np.arange(lengthy)
    data7 = np.zeros([lengthy, lengthx])

    # Fill the with data
    # Inventory Upstream
    # for x in itertools.product(y_axis, x_axis):
    #     inv_1 = max(x[1], 0)
    #     bo_1 = abs(min(x[1], 0))
    #     inv_0 = x[0]
    #     totalinventory = inv_0 + inv_1
    #     totalbo = bo_1
    #     # RETAILER 1
    #     state1 = np.array([totalinventory+30, totalbo,  # Total inventory and total backorders
    #                     inv_0,inv_1,15,15,              # Inventory per stockpoint
    #                     bo_1,0,0,                       # Backorders per stockpoint
    #                     30,10,10,10])
    #     o1 = torch.as_tensor([scale_input(env, state1)], dtype = torch.float32)
    #     a1, _, _, _, stdev = ac.step(o1, True)
    #     a1 = check_action_space(a1[0])
    #     # RETAILER 2
    #     state2 = np.array([totalinventory+30, totalbo,  # Total inventory and total backorders
    #                     inv_0,15,inv_1,15,              # Inventory per stockpoint
    #                     0,bo_1,0,                       # Backorders per stockpoint
    #                     30,10,10,10])
    #     o2 = torch.as_tensor([scale_input(env, state2)], dtype = torch.float32)
    #     a2, _, _, _, stdev = ac.step(o2, True)
    #     a2 = check_action_space(a2[0])
    #     # RETAILER 3
    #     state3 = np.array([totalinventory+30, totalbo,                # Total inventory and total backorders
    #                     inv_0,15,15,inv_1,    # Inventory per stockpoint
    #                     0,0,bo_1,                  # Backorders per stockpoint
    #                     30,10,10,10])
    #     o3 = torch.as_tensor([scale_input(env, state3)], dtype = torch.float32)
    #     a3, _, _, _, stdev = ac.step(o3, True)
    #     a3 = check_action_space(a3[0])
    #     # Add data
    #     data[int(x[0]), int((x[1]+(lengthx/2)))]  = int(a1[1])
    #     data2[int(x[0]), int((x[1]+(lengthx/2)))] = int(a2[2])
    #     data3[int(x[0]), int((x[1]+(lengthx/2)))] = int(a3[3])

    # # In Transit
    # for x2 in itertools.product(y_axis2, x_axis):
    #     in_transit  = x2[0]
    #     inv_1 = max(x2[1], 0)
    #     bo_1 = abs(min(x2[1], 0))
    #     inv_0 = 30
    #     totalinventory = inv_0 + inv_1
    #     totalbo = bo_1
    #     # RETAILER 1
    #     state1 = np.array([totalinventory+30, totalbo,  # Total inventory and total backorders
    #                     inv_0,inv_1,15,15,              # Inventory per stockpoint
    #                     bo_1,0,0,                       # Backorders per stockpoint
    #                     30,in_transit,10,10])
    #     o1 = torch.as_tensor([scale_input(env, state1)], dtype = torch.float32)
    #     a1, _, _, _, stdev = ac.step(o1, True)
    #     a1 = check_action_space(a1[0])
    #     # RETAILER 2
    #     state2 = np.array([totalinventory+30, totalbo,  # Total inventory and total backorders
    #                     inv_0,15,inv_1,15,              # Inventory per stockpoint
    #                     0,bo_1,0,                       # Backorders per stockpoint
    #                     30,10,in_transit,10])
    #     o2 = torch.as_tensor([scale_input(env, state2)], dtype = torch.float32)
    #     a2, _, _, _, stdev = ac.step(o2, True)
    #     a2 = check_action_space(a2[0])
    #     # RETAILER 3
    #     state3 = np.array([totalinventory+30, totalbo,                # Total inventory and total backorders
    #                     inv_0,15,15,inv_1,    # Inventory per stockpoint
    #                     0,0,bo_1,                  # Backorders per stockpoint
    #                     30,10,10,in_transit])
    #     o3 = torch.as_tensor([scale_input(env, state3)], dtype = torch.float32)
    #     a3, _, _, _, stdev = ac.step(o3, True)
    #     a3 = check_action_space(a3[0])
    #     # Add data
    #     data4[int(x2[0]), int((x2[1]+(lengthx/2)))]  = int(a1[1])
    #     data5[int(x2[0]), int((x2[1]+(lengthx/2)))] = int(a2[2])
    #     data6[int(x2[0]), int((x2[1]+(lengthx/2)))] = int(a3[3])

    # for x3 in itertools.product(y_axis_warehouse, x_axis_warehouse):
    #     in_transit  = x3[0]
    #     inv_0 = x3[1]
    #     totalinventory = inv_0 + 45
    #     # Warehouse
    #     state_warehouse = np.array([totalinventory, 0,  # Total inventory and total backorders
    #                     inv_0,15,15,15,              # Inventory per stockpoint
    #                     0,0,0,                       # Backorders per stockpoint
    #                     in_transit,10,10,10])
    #     o1 = torch.as_tensor([scale_input(env, state_warehouse)], dtype = torch.float32)
    #     a1, _, _, _, stdev = ac.step(o1, True)
    #     a1 = check_action_space(a1[0])
    #     # Add data
    #     data7[int(x3[0]), int(x3[1])]  = int(a1[0])

    # vmin = min(np.amin(data), np.amin(data2), np.amin(data3), np.amin(data4), np.amin(data5), np.amin(data6))
    # vmax = myround(max(np.amax(data), np.amax(data2), np.amax(data3), np.amax(data4), np.amax(data5), np.amax(data6)))
    # df1 = pd.DataFrame(data, columns=x_axis, index=y_axis)
    # df2 = pd.DataFrame(data2, columns=x_axis, index=y_axis)
    # df3 = pd.DataFrame(data3, columns=x_axis, index=y_axis)
    # df4 = pd.DataFrame(data4, columns=x_axis, index=y_axis2)
    # df5 = pd.DataFrame(data5, columns=x_axis, index=y_axis2)
    # df6 = pd.DataFrame(data6, columns=x_axis, index=y_axis2)
    # df7 = pd.DataFrame(data7, columns=x_axis_warehouse, index=y_axis_warehouse)

    # vmax_w = myround(np.amax(data7))
    # plot_actions_heatmap_warehouse_contour(df7, 0, vmax_w)
    # plot_actions_heatmap_contour([df1, df2, df3, df4, df5, df6], vmin, vmax)

    return df, holdinglist, bolist, rewardlist


def get_benchmark_results(env, rn, length, df, action):
    random.seed(rn)
    np.random.seed(rn)
    _ = env.reset()
    rewardlist, inventorylist, backorderlist = [], [], []
    for t in range(case.horizon):
        _, reward, _, info = env.simulate(action)
        if t >= case.warmup:
            inventorylist.append(info['holding_costs'])
            backorderlist.append(info['backorder_costs'])
            rewardlist.append(reward)
            df = df._append({'RN': rn,
                            'Benchmark_Inventory': info['holding_costs'],
                            'Benchmark_Backorders': info['backorder_costs'],
                            'Benchmark_Costs': reward}, ignore_index=True)

    totalfulfilled = env.TotalFulfilled
    totaldemand = env.TotalDemand
    totalBO = env.TotalBO
    fillrate = totalfulfilled / totaldemand
    print("BENCH")
    print(fillrate)
    return df, inventorylist, backorderlist, rewardlist


def plot_actions_heatmap(data, vmin, vmax):
    fig = plt.figure(figsize=(9, 6))
    bottom, top, left, right = 0.2, 0.9, 0.1, 0.85
    fig.subplots_adjust(bottom=bottom, left=left, right=right, top=top)
    fig, axn = plt.subplots(2, 3, sharex=True, sharey=True)
    cbar_ax = fig.add_axes([.91, .3, .03, .4])
    for i, ax in enumerate(axn.flat):
        sns.heatmap(data[i], ax=ax, cbar=i == 0, vmin=vmin, vmax=vmax, xticklabels=50, yticklabels=10,
                    cbar_ax=None if i else cbar_ax, cmap='gist_gray_r')
        ax.invert_yaxis()
        if i == 0:
            ax.set_ylabel(r"Inventory Warehouse", size=10)
        if i == 3:
            ax.set_ylabel(r"In Transit", size=10)
        if i >= 3:
            ax.set_xlabel("Inventory retailer {}".format(i - 2), size=10)
    # ylabel only on the left
    plt.tight_layout(rect=[0, 0, .9, 1])
    plt.savefig(directory + "{} - plot.png".format(rn))
    # tikzplotlib.save(directory + "mytikz.tex")

def plot_actions_heatmap_contour(data, vmin, vmax):
    fig = plt.figure(figsize=(9, 6))
    bottom, top, left, right = 0.2, 0.9, 0.1, 0.85
    fig.subplots_adjust(bottom=bottom, left=left, right=right, top=top)
    fig, axn = plt.subplots(2, 3, sharex=True, sharey=True)
    
    for i, ax in enumerate(axn.flat):
        smooth_scale = 5
        z = ndimage.zoom(data[i].to_numpy(), smooth_scale)
        cntr = ax.contourf(np.linspace(0, len(data[i].columns), len(data[i].columns) * smooth_scale),
                   np.linspace(0, len(data[i].index), len(data[i].index) * smooth_scale),
                   z, levels=np.arange(0, 50, 5), cmap='gist_gray_r')
        sns.heatmap(data[i], ax=ax, vmin=vmin, vmax=vmax, xticklabels=50, yticklabels=10, alpha = 0,
                    cbar_ax = None, cbar = False)
        ax.invert_yaxis()
        
        if i == 0:
            ax.set_ylabel(r"Inventory Warehouse", size=10)
        if i == 3:
            ax.set_ylabel(r"In Transit", size=10)
        if i >= 3:
            ax.set_xlabel("Inventory retailer {}".format(i - 2), size=10)
    # ylabel only on the left
    cbar_ax = fig.add_axes([.91, .1, .03, .8])
    plt.colorbar(cntr, cax = cbar_ax)#, location = 'right')
    plt.tight_layout(rect=[0, 0, .9, 1])
    plt.savefig(directory + "{} - plot.png".format(rn))

def plot_actions_heatmap_warehouse(data, vmin, vmax):
    fig = plt.figure()
    ax = sns.heatmap(data, vmin=0, vmax=vmax, xticklabels=50, yticklabels=10, cmap='gist_gray_r')
    ax.invert_yaxis()
    ax.set_xlabel(r"Inventory Warehouse", size=10)
    ax.set_ylabel(r"In Transit", size=10)
    # ylabel only on the left
    plt.tight_layout(rect=[0, 0, .9, 1])
    plt.savefig(directory + "{} - plot warehouse.png".format(rn))
    # tikzplotlib.save(directory + "mytikz.tex")

def plot_actions_heatmap_warehouse_contour(data, vmin, vmax):
    fig, ax = plt.subplots()
    smooth_scale = 5
    z = ndimage.zoom(data.to_numpy(), smooth_scale)
    cntr = ax.contourf(np.linspace(0, len(data.columns), len(data.columns) * smooth_scale),
                   np.linspace(0, len(data.index), len(data.index) * smooth_scale),
                   z, levels=np.arange(0, 70, 5), cmap='gist_gray_r')
    ax = sns.heatmap(data, vmin=0, vmax=vmax, xticklabels=50, yticklabels=10, alpha = 0, cbar = False, ax = ax)
    ax.invert_yaxis()
    plt.colorbar(cntr, ax = ax)
    ax.set_xlabel(r"Inventory Warehouse", size=10)
    ax.set_ylabel(r"In Transit", size=10)
    # ylabel only on the left
    plt.tight_layout(rect=[0, 0, .9, 1])
    plt.savefig(directory + "{} - plot warehouse.png".format(rn))
    # tikzplotlib.save(directory + "mytikz.tex")

def plot_actions_surface(x, y, z):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the surface
    ax.plot_trisurf(x, y, z, linewidth=0.2, antialiased=True)

    plt.savefig(directory + "plotsurface.png")


def plot_actions_scatter(x, y, data, hue):
    plt.figure(figsize=(16, 10))
    fig = sns.scatterplot(x=x, y=y, data=data, hue=hue)
    plt.savefig(directory + "plot.png", dpi=1000)


def plot_costs_over_time(prod1, prod2, costs, title, start, end):
    plt.figure(figsize=(8, 5))
    plt.plot(prod1[start:end], label='Inventory', linewidth=2)
    plt.plot(prod2[start:end], label='Backorders', linewidth=2)
    plt.title(title + str(costs), fontsize=18)
    plt.ylabel('inventory level')
    plt.ylim(bottom=0, top=300)
    plt.legend()
    fig_title = directory + title + str(end) + 'inv'
    plt.savefig(fig_title, dpi=1000)


# Plot acties voor een given state als input
# def plot_action_distribution(mean, std, x_min, x_max, title):
#     fig, axs = plt.subplots(2, 2)
#     k = 0
#     for i in range(2):
#         for j in range(2):
#             x = np.linspace(x_min[k], x_max[k], (x_max[k]-x_min[k])*100)
#             y = scipy.stats.norm.pdf(x,mean[k],std[k])
#             axs[i, j].plot(x,y)
#             plt.xlim(x_min[k],x_max[k])
#             plt.ylim(0,1.5)
#             axs[i, j].set_title('Action ' + str(k))
#             k += 1
#     fig_title = directory + title + 'dist'
#     plt.savefig(fig_title, dpi = 1000)
#     plt.clf()


# TODO: Add In Transit and Warehouse graphs to 1 graph
# TODO: Simulated data to df
ppo_df = pd.DataFrame({'RN': [],
                       'Benchmark_Inventory': [],
                       'Benchmark_Backorders': [],
                       'Benchmark_Costs': [],
                       'PPO_Inventory': [],
                       'PPO_Backorders': [],
                       'PPO_Costs': []})

benchmark_df = pd.DataFrame({'RN': [],
                             'Benchmark_Inventory': [],
                             'Benchmark_Backorders': [],
                             'Benchmark_Costs': [],
                             'PPO_Inventory': [],
                             'PPO_Backorders': [],
                             'PPO_Costs': []})

env2 = InventoryEnv(case2, case.action_low, case.action_high, case.action_min, case.action_max,
                    case.state_low, case.state_high)


benchmark_df, benchmark_inventory, benchmark_backorders, benchmark_reward = get_benchmark_results(env2, 1, 200,benchmark_df,basestock_action)
ppo_df, ppo_inventory, ppo_backorders, ppo_reward = get_ppo_policy_results(env, 1, 200, ppo_df,
                                                                               'RN1/pyt_network_save/modelsave0.pt')
save_states_encountered(env,0, 2000, 'RN1/pyt_network_save/modelsave0.pt')
df = ppo_df.combine_first(benchmark_df)
#     # plot_costs_over_time(benchmark_inventory, benchmark_backorders,
#     #                 np.sum(benchmark_reward), '{} - Benchmark policy over time'.format(rn), 50, 100)
#     # plot_costs_over_time(ppo_inventory, ppo_backorders,
#     #                 np.sum(ppo_reward), '{} - PPO policy over time'.format(rn), 50, 100)
df.to_csv(directory + 'data1.csv', sep=";", decimal=",")
