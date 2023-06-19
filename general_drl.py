"""@author: KevinG."""
from inventory_env import InventoryEnv
from ppo.ppo_training_functions import ppo_learning
from ppo.ppo_support_functions import set_seeds
from cases import General
import ppo_settings

experiment_name = 'CBC/CAPtestechtv2/'
case = General()

# length of simulation to compute benchmark and as stopping criterion
ppo_simulation_length = case.horizon
# length of initial simulation that is discarded
ppo_warmup_period = case.warmup

policy_results_states = [[0, 12, 12, 12, 12]]

for k in range(1,2):
    print("Replication " + str(k))
    # Initialize environment
    env = InventoryEnv(case, case.action_low, case.action_high,
                       case.action_min, case.action_max, case.state_low, case.state_high)
    run_name = "RN{}".format(k)

    # set random seed
    set_seeds(env, k)

    # call learning function
    ppo_learning(env, False, experiment_name, run_name,
                 ppo_settings.network_activation, ppo_settings.network_size, ppo_settings.network_bias_init,
                 ppo_settings.network_weights_init, ppo_settings.ppo_evaluation_steps,
                 ppo_settings.ppo_evaluation_threshold, ppo_settings.ppo_iterations, ppo_settings.ppo_buffer_length,
                 ppo_settings.ppo_gamma, ppo_settings.ppo_lambda, ppo_settings.cooldown_buffer,
                 ppo_settings.ppo_epsilon, ppo_settings.pi_lr, ppo_settings.vf_lr, ppo_settings.ppo_save_freq,
                 ppo_settings.ppo_epochs, ppo_settings.ppo_batch_size, ppo_settings.ppo_simulation_runs,
                 ppo_simulation_length, ppo_warmup_period, policy_results_states)
