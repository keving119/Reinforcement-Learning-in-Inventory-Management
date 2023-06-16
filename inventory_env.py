"""@author: KevinG."""
import random
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import gym
from gym.spaces import Box, Dict


# Function to convert named state space to a flattened input array
def named_state_space_to_input(named_state_space):
    print(named_state_space)
    input_array = np.array([], dtype=int)
    for key in named_state_space:
        input_array = np.concatenate((input_array, np.array(named_state_space[key]).flatten()))
    print(input_array.shape)
    return input_array

def generate_leadtime(t, dist, lowerbound, upperbound):
    """
    Generate the leadtime of the dataset from paper or distribution.

    Returns: Integer
    """
    if dist == 'uniform':
        leadtime = random.randrange(lowerbound, upperbound + 1)
    else:
        raise Exception
    return leadtime

# Function to create an empty dictionary with keys for each lead time
def create_lead_time_dict(total_stockpoints_connections, max_lead_time):
    lt_dict = {}
    for lt in range(max_lead_time):
        lt_dict[f'lt{lt}'] = [0] * (total_stockpoints_connections)
    return lt_dict

# Function to create an empty dictionary with keys for each lead time
def create_lead_time_state(state_space, max_lead_time):
    lt_dict = {}
    for lt in range(max_lead_time):
        lt_dict[f'lt{lt}'] = state_space
    return lt_dict

class InventoryEnv(gym.Env):
    """
    General Inventory Control Environment.

    Currently tested with:
    - A reinforcement learning model for supply chain ordering management:
      An application to the beer game - Chaharsooghi (2002)
    - Kunnumkal (2002)
    """

    def __init__(self, case, action_low, action_high, action_min, action_max,
                 state_low, state_high):
        self.case = case
        self.case_name = case.__class__.__name__
        self.n = case.leadtime_ub + 1   # Horizon is defined by maximum leadtime
        self.action_low = action_low
        self.action_high = action_high
        self.action_min = action_min
        self.action_max = action_max
        self.state_low = state_low
        self.state_high = state_high
        self.determine_potential_actions()
        self.determine_potential_states()

    def determine_potential_actions(self):
        """
        Possible actions returned as Gym Space
        each period
        """
        self.feasible_actions = 0
        self.action_space = Box(self.action_low, self.action_high, dtype=np.int32)

    def determine_potential_states(self):
        """
        Based on the mean demand, we determine the maximum and minimum
        inventory to prevent the environment from reaching unlikely states
        """
        self.observation_space = Box(self.state_low, self.state_high, dtype=np.int32)

        self.observation_space = Dict({
            'total_inventory': Box(0, self.state.high, dtype=np.int32),
            'total_backorders': Box(0, self.state.high, dtype=np.int32),
            'inventory': Box(0, self.state.high, dtype=np.int32),
            'backorders': Box(0, self.state.high, dtype=np.int32),
            'in_transit': create_lead_time_state(Box(0, [self.state_in_transit] * self.case.no_nodes, dtype=np.int32), self.case.leadtime_ub)
        })

    def _generate_demand(self):
        """
        Generate the demand using a predefined distribution.
        Writes the demand to the orders table.
        """
        source, destination = np.nonzero(self.case.connections)
        for retailer, customer in zip(source[-self.case.no_customers:],
                                      destination[-self.case.no_customers:]):
            if self.case.demand_dist == 'poisson':
                demand_mean = random.randrange(self.case.demand_lb,
                                               self.case.demand_ub + 1)
                demand = np.random.poisson(demand_mean)
            elif self.case.demand_dist == 'uniform':
                demand = random.randrange(self.case.demand_lb,
                                          self.case.demand_ub + 1)
            self.O[0, customer, retailer] = demand
            if (self.t < self.case.horizon) and (self.t >= self.case.warmup): 
                self.TotalDemand[customer,retailer] += demand

    def calculate_reward(self):
        """
        Calculate the reward for the current period.

        Returns: holding costs, backorder costs
        """
        backorder_costs = np.sum(self.BO[0] * self.case.bo_costs)
        holding_costs = np.sum(self.INV[0] * self.case.holding_costs)
        return holding_costs, backorder_costs

    def _initialize_state(self):
        """
        Initialize the inventory position for every node.

        Copies the inventory position from the previous timestep.
        """
        for i in range(self.n-1):
            self.T[i]           = np.copy(self.T[i+1])
            self.O[i]           = np.copy(self.O[i+1])
            self.in_transit[i]  = np.copy(self.in_transit[i+1])
        self.T[self.n-1]          = 0
        self.O[self.n-1]          = 0
        self.in_transit[self.n-1] = 0

    def _receive_incoming_delivery(self):
        """
        Receives the incoming delivery for every stockpoint.

        Customers are not taken into account because of zero lead time
        Based on the amount stated in T
        """
        # Loop over all suppliers and stockpoints
        for i in range(0, self.case.no_stockpoints + self.case.no_suppliers):
            # Loop over all stockpoints
            # Note that only forward delivery is possible, hence 'i+1'
            for j in range(i + 1, self.case.no_stockpoints +
                           self.case.no_suppliers):
                delivery = self.T[0, i, j]
                self.INV[0, j] += delivery
                self.in_transit[0, i, j] -= delivery
                self.T[0, i, j] = 0

    def _receive_incoming_orders(self):
        # Loop over every stockpoint
        for i in range(self.case.no_stockpoints + self.case.no_suppliers):
            # Check if the inventory is larger than all incoming orders
            if self.INV[0, i] >= np.sum(self.O[0, :, i], 0):
                # print(f"Inventory of {i} is {self.INV[0, i]} which is larger than {np.sum(self.O[0, :, i], 0)}")
                for j in np.nonzero(self.case.connections[i])[0]:
                    if self.O[0, j, i] > 0:
                        self._fulfill_order(i, j, self.O[0, j, i])
                        if self.t >= self.case.warmup:
                            self.TotalFulfilled[j,i] += self.O[0,j,i]
            else:
                IPlist = {}
                # print(f"Inventory of {i} is {self.INV[0, i]} which is smaller than {np.sum(self.O[0, :, i], 0)}")
                # Generate a list of stockpoints that have outstanding orders
                k_list = np.nonzero(self.O[0, :, i])[0]
                bo_echelon = np.sum(self.BO[0], 0)
                for k in k_list:
                    # Would be more logical to include backorders at retailer and In transit items
                    IPlist[k] = self.INV[0, k] - bo_echelon[k]
                # Check the lowest inventory position and sort these on lowest IP
                sorted_IP = {k: v for k, v in sorted(IPlist.items(), key=lambda item: item[1])}
                for j in sorted_IP:
                    inventory = self.INV[0, i]
                    # Check if the remaining order can be fulfilled completely
                    if inventory >= self.O[0, j, i]:
                        self._fulfill_order(i, j, self.O[0, j, i])
                        if self.t >= self.case.warmup:
                            self.TotalFulfilled[j,i] += self.O[0,j,i]
                    else:
                    # Else, fulfill how far possible
                        quantity = self.O[0, j, i] - inventory
                        self._fulfill_order(i, j, inventory)
                        if self.t >= self.case.warmup:
                            self.TotalFulfilled[j,i] += inventory
                        if self.case.unsatisfied_demand == 'backorders':
                            # print(f"Not enough inventory, backorders will be added. stockpoint{i} to {j}, inventory of i:{inventory}, Backorder prev:{self.BO[0,j,i]}, bo quantity {quantity}")
                            self.BO[0, j, i] += quantity
                            if self.t >= self.case.warmup:
                                # print("i: {} j: {} inventory:{}".format(i, j, inventory))
                                self.TotalBO[j,i] += quantity
        if self.case.unsatisfied_demand == 'backorders':
            i_list, j_list = np.nonzero(self.case.connections)
            for i, j in zip(i_list, j_list):
                inventory = self.INV[0, i]
                backorder = self.BO[0, j, i]
                # If there are any backorders, fulfill them afterwards
                if inventory > 0 and backorder > 0:
                    # If the inventory is larger than the backorder
                    # Fulfill the whole backorder
                    if inventory >= backorder:
                        self._fulfill_order(i, j, backorder)
                        # print(f"backorder will be fullfilled. stockpoint{i} to {j}, inventory of i:{inventory}, Backorder prev:{self.BO[0,j,i]}, backorder:{backorder}")
                        self.BO[0, j, i] = 0
                    # Else, fulfill the entire inventory
                    else:
                        self._fulfill_order(i, j, inventory)
                        # print(f"inventory will be fullfilled to reduce backorders. stockpoint{i} to {j}, inventory of i:{inventory}, Backorder prev:{self.BO[0,j,i]}")
                        self.BO[0, j, i] -= inventory

    def _recieve_incoming_orders_customers(self):
        i_list, j_list = np.nonzero(self.case.connections)
        for i, j in zip(i_list[-self.case.no_customers:], j_list[-self.case.no_customers:]):
            if self.O[0, j, i] > 0:
                # Check if the current order can be fulfilled
                if self.INV[0, i] >= self.O[0, j, i]:
                    self._fulfill_order(i, j, self.O[0, j, i])
                    # Else, fulfill as far as possible
                else:
                    inventory = max(self.INV[0, i], 0)
                    quantity = self.O[0, j, i] - inventory
                    self._fulfill_order(i, j, inventory)
                    # Add to backorder if applicable
                    if self.case.unsatisfied_demand == 'backorders':
                        # print(f"Not enough inventory, backorders will be added. stockpoint{i} to {j}, inventory of i:{inventory}, Backorder prev:{self.BO[0,j,i]}, bo quantity {quantity}")
                        self.BO[0, j, i] += quantity
        if self.case.unsatisfied_demand == 'backorders':
            for i, j in zip(i_list[-self.case.no_customers:], j_list[-self.case.no_customers:]):
                inventory = self.INV[0, i]
                # If there are any backorders, fulfill them afterwards
                if inventory > 0 and backorder > 0:
                    # If the inventory is larger than the backorder
                    # Fulfill the whole backorder
                    backorder = self.BO[0, j, i]
                    if inventory >= backorder:
                        self._fulfill_order(i, j, backorder)
                        # print(f"backorder will be fullfilled. stockpoint{i} to {j}, inventory of i:{inventory}, Backorder prev:{self.BO[0,j,i]}, backorder:{backorder}")
                        self.BO[0, j, i] = 0
                    # Else, fulfill the entire inventory
                    else:
                        self._fulfill_order(i, j, inventory)
                        # print(f"inventory will be fullfilled to reduce backorders. stockpoint{i} to {j}, inventory of i:{inventory}, Backorder prev:{self.BO[0,j,i]}")
                        self.BO[0, j, i] -= inventory

    def _recieve_incoming_orders_divergent(self):
        # Ship from supplier to warehouse
        self._fulfill_order(0, 1, self.O[0, 1, 0])
        # Check if the warehouse can ship all orders
        if self.INV[0, 1] >= np.sum(self.O[0, :, 1], 0):
            i_list, j_list = np.nonzero(self.case.connections)
            for i, j in zip(i_list[self.case.no_suppliers:self.case.no_suppliers+
                                   self.case.no_stockpoints],
                            j_list[self.case.no_suppliers:self.case.no_suppliers+
                                   self.case.no_stockpoints]):
                if self.O[0, j, i] > 0:
                    self._fulfill_order(i, j, self.O[0, j, i])
        else:
            IPlist = {}
            i_list, _ = np.nonzero(self.O[0])
            bo_echelon = np.sum(self.BO[0], 0)
            for i in i_list:
                IPlist[i] = self.INV[0, i] - bo_echelon[i]
            # Check the lowest inventory position and sort these on lowest IP
            sorted_IP = {k: v for k, v in sorted(IPlist.items(), key=lambda item: item[1])}
            # Check if there is still inventory left
            if self.INV[0, 1] >= 0:
                for i in sorted_IP:
                    # Check if the remaining order can be fulfilled completely
                    if self.INV[0, 1] >= self.O[0, i, 1]:
                        self._fulfill_order(1, i, self.O[0, i, 1])
                    else:
                    # Else, fulfill how far possible
                        inventory = max(self.INV[0, 1], 0)
                        quantity = self.O[0, i, 1] - inventory
                        self._fulfill_order(1, i, inventory)
                        break

    def _fulfill_order(self, source, destination, quantity):
        # Customers don't have any lead time.
        if destination >= self.case.no_nodes - self.case.no_customers:
            leadtime = 0
        else:
            leadtime = self.leadtime
        # The order is fulfilled immediately for the customer
        # or whenever the leadtime is 0
        if leadtime == 0:
            # The new inventorylevel is increased with the shipped quantity
            self.INV[0, destination] += quantity
        else:
            # If the order is not fulfilled immediately, denote the time when
            # the order will be delivered. This can not be larger than the horizon
            if leadtime < self.n:
                self.T[leadtime, source, destination] += quantity
            else:
                raise NotImplementedError
            for k in range(0, min(leadtime, self.n) + 1):
                self.in_transit[k, source, destination] += quantity
        # Suppliers have unlimited capacity
        if source >= self.case.no_suppliers:
            self.INV[0, source] -= quantity

    def _place_outgoing_order(self, t, action):
        k = 0
        incomingOrders = np.sum(self.O[0], 0)
        # Loop over all suppliers and stockpoints
        for j in range(self.case.no_suppliers, self.case.no_stockpoints +
                        self.case.no_suppliers):
            RandomNumber = random.random()
            probability = 0
            for i in range(0, self.case.no_stockpoints + self.case.no_suppliers):
                if self.case.connections[i, j] == 1:
                    self._place_order(i,j,t,k, action, incomingOrders)
                    k += 1
                elif self.case.connections[i,j] > 0:
                    probability += self.case.connections[i,j]
                    if RandomNumber < probability:
                        self._place_order(i,j,t,k, action, incomingOrders)
                        k += 1
                        break

    def _place_order(self, i, j, t, k, action, incomingOrders):
        if self.case.order_policy == 'X':
            # print("K:{}, i:{}, j:{}".format(k, i, j))
            # print(action)
            self.O[t, j, i] += action[k]
            if (self.t < self.case.horizon - 1) and (self.t >= self.case.warmup-1): 
                self.TotalDemand[j,i] += action[k]
        elif self.case.order_policy == 'X+Y':
            self.O[t, j, i] += incomingOrders[j] + action[k]
            if (self.t < self.case.horizon - 1) and (self.t >= self.case.warmup-1): 
                self.TotalDemand[j,i] += incomingOrders[j] + action[k]
        elif self.case.order_policy == 'BaseStock':
            bo_echelon = np.sum(self.BO[0], 0)
            if action[k] > 0:
                totalorderamount = max(0,action[k] - (self.INV[0, j] + self.in_transit[0, i, j] - bo_echelon[j]))
                number_of_upstream_connections = np.sum(self.case.connections[:, j])
                self.O[t, j, i] += max(0, totalorderamount)
                # print(f"K:{k}, i:{i}, j:{j} toa:{totalorderamount}: action:{action[k]}, inv:{self.INV[0, j]} it:{self.in_transit[0, i, j]} bo:{bo_echelon[j]}")
                # self.O[t, j, i] += max(0, totalorderamount/number_of_upstream_connections)
                if (self.t < self.case.horizon - 1) and (self.t >= self.case.warmup-1): 
                    self.TotalDemand[j,i] += max(0, totalorderamount)
                    # self.TotalDemand[j,i] += max(0, totalorderamount/number_of_upstream_connections)
        else:
            raise NotImplementedError        

    def _code_state(self):

        # Cap is nu nog niet heel netjes verwerkt en alleen voor general case.
        if self.case_name == 'General':
            # print(f"inventory:{self.INV[0, self.case.no_suppliers:-self.case.no_customers]}")
            self.INV[0, self.case.no_suppliers:-self.case.no_customers] = np.minimum(self.INV[0, self.case.no_suppliers:-self.case.no_customers], self.case.state_high[2:11])
            # print(f"inventory capped:{self.INV[0, :]}")
            # print("bo :{}".format(self.BO[0, :]))
            self.BO[0, :] =  np.minimum(self.BO[0, :], [0,0,0,0,500,500,500,500,250,250,250,250,250,0,0,0,0,0] )
            # print("bo capped:{}".format(self.BO[0, :]))

        bo_echelon = np.sum(self.BO[0], 0)
        totalinventory = np.sum(self.INV[0, self.case.no_suppliers:-self.case.no_customers], 0)
        totalbackorders = np.sum(bo_echelon, 0)

        i_list, j_list = np.nonzero(self.case.connections)

        # Create a named state space dictionary
        named_state_space = {
            'total_inventory': totalinventory,
            'total_backorders': totalbackorders,
            'inventory': [0] * (self.case.no_stockpoints),
            'backorders': [0] * (len(i_list[self.case.no_suppliers:])),
            'in_transit': create_lead_time_dict(len(i_list[:-self.case.no_customers]), self.case.leadtime_ub)
        }

        # Fill in inventory values
        for i in range(self.case.no_suppliers, self.case.no_stockpoints + self.case.no_suppliers):
            named_state_space['inventory'][i - self.case.no_suppliers] = self.INV[0, i]

        # Fill in backorders values
        k = 0
        for i, j in zip(i_list[self.case.no_suppliers:], j_list[self.case.no_suppliers:]):
            named_state_space['backorders'][k] = self.BO[0, j, i]
            k += 1

        # Fill in in_transit values
        k = 0
        for time in range(self.case.leadtime_ub):
            for i, j in zip(i_list[:-self.case.no_customers], j_list[:-self.case.no_customers]):
                named_state_space['in_transit'][f'lt{time}'][k] = self.T[time, i, j]
                k += 1

        named_state_array= named_state_space_to_input(named_state_space)

            #old
        # # This still needs to be adjusted. To implement it in the named statespace, we should allow make the named statespace more flexible to accomodate for lead times.
        # if self.case_name == 'BeerGame':
        #     in_transit0 = np.sum(self.in_transit[0], 0)
        #     in_transit1 = np.sum(self.in_transit[1], 0)
        #     in_transit2 = np.sum(self.in_transit[2], 0)
        #     in_transit3 = np.sum(self.in_transit[3], 0)
        #     in_transit4 = np.sum(self.in_transit[4], 0)
        #     transport1 = np.sum(self.T[1], 0)
        #     transport2 = np.sum(self.T[2], 0)
        #     transport3 = np.sum(self.T[3], 0)
        #     transport4 = np.sum(self.T[4], 0)
        #     for i in range(self.case.no_suppliers, self.case.no_stockpoints +
        #                 self.case.no_suppliers):
        #         CIP[i+13] = in_transit0[i]
        #         CIP[i+17] = in_transit1[i]
        #         CIP[i+21] = in_transit2[i]
        #         CIP[i+25] = in_transit3[i]
        #         CIP[i+29] = in_transit4[i]
        #         CIP[i+33] = transport1[i]
        #         CIP[i+37] = transport2[i]
        #         CIP[i+41] = transport3[i]
        #         CIP[i+45] = transport4[i]

            # elif self.case_name == 'Divergent':
            #     totalinventory = np.sum(self.INV[0, self.case.no_suppliers:-self.case.no_customers], 0)
            #     totalbackorders = np.sum(bo_echelon, 0)
            #     in_transit0 = np.sum(self.in_transit[0], 0)
            #     CIP = np.zeros([self.case.no_stockpoints*3+1], dtype=int)
            #     CIP[0] = totalinventory
            #     CIP[1] = totalbackorders
            #     for i in range(self.case.no_suppliers, self.case.no_stockpoints +
            #                     self.case.no_suppliers):
            #         CIP[i+1] = self.INV[0, i]
            #         if i > self.case.no_suppliers:
            #             CIP[i+4] = bo_echelon[i]
            #         CIP[i+8] = in_transit0[i]
            #     CIP = np.clip(CIP, self.observation_space.low, self.observation_space.high)


            # totalinventory = np.sum(self.INV[0, self.case.no_suppliers:-self.case.no_customers], 0)
            # totalbackorders = np.sum(bo_echelon, 0)
            # previousDemand = np.sum(self.O[1], 1)
            # in_transit0 = np.sum(self.in_transit[0], 0)
            # CIP = np.zeros([2+self.case.no_stockpoints+19+18], dtype=int) # [0-56]
            # CIP[0] = totalinventory
            # CIP[1] = totalbackorders
            # for i in range(self.case.no_suppliers, self.case.no_stockpoints +
            #             self.case.no_suppliers):
            #     # print(i)
            #     # print(self.INV[0, i])
            #     CIP[i - self.case.no_suppliers + 2] = self.INV[0, i]        # [2, 10]
            # # loop hier over nonzeros van connectie
            # k = 11
            # i_list, j_list = np.nonzero(self.case.connections)
            # for i, j in zip(i_list[self.case.no_suppliers:], j_list[self.case.no_suppliers:]):
            #     CIP[k] = self.BO[0, j, i]
            #     k += 1
            # for i, j in zip(i_list[:-self.case.no_customers], j_list[:-self.case.no_customers]):
            #     CIP[k] = self.in_transit[0, i, j]  
            #     k += 1
            # CIP_old = CIP
            # CIP = np.clip(CIP, self.observation_space.low, self.observation_space.high)
            # if np.array_equiv(CIP_old,CIP) == False:
                # print("Hier gaat nog iets mis")
                # print(CIP_old)
                # print(CIP)
        # CIP = np.clip(CIP, self.observation_space.low, self.observation_space.high)
        return named_state_array
    
    def _check_action_space(self, action):
        if isinstance(self.action_space, spaces.Box):
            low = self.action_space.low
            high = self.action_space.high
            max = self.action_max
            min = self.action_min
            action_clip = np.clip(action, low, high)
            penalty = 0
            for i in range(len(action_clip)):
                action_clip[i] = ((action_clip[i] - low[i]) / (high[i]-low[i])) * ((max[i] - min[i])) + min[i]
            action = [np.round(num) for num in action_clip]
        else:
            penalty = 0  
        return action, penalty
                 
    def step(self, action, visualize=False):
        """
        Execute one step in the RL method.

        input: actionlist, visualize
        """
        self.leadtime = generate_leadtime(0, self.case.leadtime_dist,
                                          self.case.leadtime_lb, self.case.leadtime_ub)
        action, penalty = self._check_action_space(action)
        self._initialize_state()
        if visualize: self._visualize("0. IP")
        if self.case_name == "BeerGame" or self.case_name == "General":
            self._generate_demand()
            self._receive_incoming_delivery()
            if visualize: self._visualize("1. Delivery")
            self._receive_incoming_orders()
            if visualize: self._visualize("2. Demand")
            self._place_outgoing_order(1, action)
        elif self.case_name == "Divergent":
            # According to the paper:
            # (1) Warehouse places order to external supplier
            self._place_outgoing_order(0, action)
            if visualize: self._visualize("1. Warehouse order")
            # (2) Warehouse ships the orders to retailers taking the inventory position into account
            self._recieve_incoming_orders_divergent()
            if visualize: self._visualize("2. Warehouse ships")
            # (3) Warehouse and retailers receive their orders
            self._receive_incoming_delivery()
            if visualize: self._visualize("3. Orders received")
            # (4) Demand from customers is observed
            self._generate_demand()
            self._recieve_incoming_orders_customers()    
            if visualize: self._visualize("4. Demand")
        else:
            raise NotImplementedError
        CIP = self._code_state()
        holding_costs, backorder_costs = self.calculate_reward()
        reward = holding_costs + backorder_costs + penalty
        return CIP, -reward/self.case.divide, False,  {'holding_costs':holding_costs,'backorder_costs':backorder_costs, 'penalty_costs':penalty}

    def simulate(self, action, visualize=False):
        """
        Execute one step in the RL method.
        Uses predefined datasets
        input: actionlist, visualize
        """
        action, _ = self._check_action_space(action)
        self.leadtime = generate_leadtime(self.t, self.case.leadtime_dist, self.case.leadtime_lb, self.case.leadtime_ub)
        self._initialize_state()
        if visualize: self._visualize("0. IP")
        if self.case_name == "BeerGame" or self.case_name == "General":
            self._generate_demand()
            self._receive_incoming_delivery()
            if visualize: self._visualize("1. Delivery")
            self._receive_incoming_orders()
            if visualize: self._visualize("2. Demand")
            self._place_outgoing_order(1, action)
        elif self.case_name == "Divergent":
            # According to the paper:
            # (1) Warehouse places order to external supplier
            self._place_outgoing_order(0, action)
            if visualize: self._visualize("1. Warehouse order")
            # (2) Warehouse ships the orders to retailers taking the inventory position into account
            self._recieve_incoming_orders_divergent()
            if visualize: self._visualize("2. Warehouse ships")
            # (3) Warehouse and retailers receive their orders
            self._receive_incoming_delivery()
            if visualize: self._visualize("3. Orders received")
            # (4) Demand from customers is observed
            self._generate_demand()
            self._recieve_incoming_orders_customers()    
            if visualize: self._visualize("4. Demand")
        else:
            raise NotImplementedError
        CIP = self._code_state()
        holding_costs, backorder_costs = self.calculate_reward()
        reward = holding_costs + backorder_costs
        self.t += 1
        if self.t == self.case.horizon: 
            self.t = 0
        return CIP, -reward, False,  {'holding_costs':holding_costs,'backorder_costs':backorder_costs}

    def reset(self):
        """
        Reset the simulation.

        Has to be executed at the beginning of every iteration.
        """
        self.t = 0
        # Amount of inventory of stockpoint s in time t
        self.INV = np.zeros([self.n, self.case.no_nodes], dtype=int)
        # Amount of backorders of stockpoint s+1 to stockpoint s in time t
        self.BO = np.zeros([self.n, self.case.no_nodes, self.case.no_nodes], dtype=int)
        # Set the initial inventory level as given for every stockpoint
        self.INV[0] = self.case.initial_inventory
        # Inventory position (inventory-backorders) of echelon s in time t
        self.IP = np.copy(self.INV)
        # Number of items ordered from stockpoint s to stockpoint s-1
        # An order from stockpoint s to s-1, is an order that is placed upstream
        self.O = np.copy(self.BO)
        # Number of items arriving in time t from stockpoint s to stockpoint s+1
        self.T = np.copy(self.BO)
        # Number of items in transit from stockpoint s to stockpoint s+1
        self.in_transit = np.copy(self.BO)
        # Total demand from stockpoint s to stockpoint s+1
        self.TotalDemand = np.zeros([self.case.no_nodes, self.case.no_nodes], dtype=int)
        # Total fulfilled from stockpoint s to stockpoint s+1
        self.TotalFulfilled = np.copy(self.TotalDemand)
        self.TotalBO = np.copy(self.TotalDemand)
        # Custom BeerGame Settings.
        if self.case_name == "BeerGame":
            self.O[2, 1, 0] = 4
            self.O[2, 2, 1] = 4
            self.O[2, 3, 2] = 4
            self.in_transit[0, 3, 4] = 8
            self.in_transit[1, 3, 4] = 8
            self.in_transit[2, 3, 4] = 4
            self.in_transit[0, 2, 3] = 4
            self.in_transit[1, 2, 3] = 4
            self.in_transit[2, 2, 3] = 4
            self.in_transit[0, 1, 2] = 4
            self.in_transit[1, 1, 2] = 4
            self.in_transit[2, 1, 2] = 4
            self.in_transit[0, 0, 1] = 4
            self.in_transit[1, 0, 1] = 4
            self.in_transit[2, 0, 1] = 4
            self.T[1, 3, 4] = 4
            self.T[2, 3, 4] = 4
            self.T[2, 2, 3] = 4
            self.T[2, 1, 2] = 4
            self.T[2, 0, 1] = 4
        self.state = self._code_state()
        return self.state

    def _visualize(self, action):
        # Initialize graph
        Graph = nx.DiGraph()
        i = 0
        label, pos, pos_edges = {}, {}, {}
        # TODO: Deze positionlist logisch maken
        postionlist = [0, 1, -1, 2, -2, 3, -3, 4, -4, 5, -5]
        for echelon in range(self.case.no_echelons):
            for stockpoint in range(self.case.stockpoints_echelon[echelon]):
                Graph.add_node(i)
                label[i] = self.INV[0, i]
                pos[i] = [echelon, 0.1*postionlist[stockpoint]]
                i += 1
        nx.draw(Graph, pos, edgelist=[])
        nx.draw_networkx_labels(Graph, pos, labels=label)

        # Draw edges for 'order_size' (T)
        for source in range(self.case.no_nodes):
            for destination in range(self.case.no_nodes):
                in_transit = self.in_transit[0, source, destination]
                if self.T[0, source, destination] > 0:
                    Graph.add_edge(source, destination,
                                   order_size='T:' + str(self.T[0, source, destination]) + ' IT:' + str(in_transit))
                elif self.case.connections[source][destination] > 0:
                    Graph.add_edge(source, destination, order_size='T:' + str(0) + ' IT:' + str(in_transit))

        draw_labels(Graph, pos, pos_edges, 'order_size', 0, 0.005)

        # draw edges for 'in_order' (O)
        for source in range(self.case.no_nodes):
            for destination in range(self.case.no_nodes):
                backorder = self.BO[0, source, destination]
                if self.O[0, source, destination] > 0:
                    Graph.add_edge(source, destination,
                                   in_order='O:' + str(self.O[0, source, destination]) + ' BO:' + str(backorder))
                elif self.case.connections[destination][source] > 0:
                    Graph.add_edge(source, destination, in_order='O:' + str(0) + ' BO:' + str(backorder))
        nx.draw_networkx_edges(Graph, pos)

        draw_labels(Graph, pos, pos_edges, 'in_order', 0, -0.005)

        plt.suptitle('Time: ' + str(self.t) + ' Action: ' + action, fontsize=11)
        plt.savefig(str(self.t)+action+'.png')
        plt.show()
        Graph.clear()

def draw_edges(graph, nodes, name1, table1, name2, table2, connections, defaultorder):
    for source in range(nodes):
        for destination in range(nodes):
            table2_value = table2[0, source, destination]
            if table1[0, source, destination] > 0:
                graph.add_edge(source, destination, name1=table1[0, source, destination], name2=table2_value)
            elif defaultorder:
                if connections[source, destination] > 0:
                    graph.add_edge(source, destination, name1=0, name2=table2_value)
            elif not defaultorder:
                if connections[destination, source] > 0:
                    graph.add_edge(source, destination, name1=0, name2=table2_value)


def draw_labels(graph, pos, pos_edges, name, echelon_pos, height_pos):
    for p in pos:
        echelon = pos[p][0]
        height = pos[p][1]
        pos_edges[p] = [echelon + echelon_pos, height + height_pos]
    labels = nx.get_edge_attributes(graph, name)
    nx.draw_networkx_edge_labels(graph, pos_edges, edge_labels=labels)
