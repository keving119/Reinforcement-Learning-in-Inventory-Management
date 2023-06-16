import numpy as np


class BeerGame:
    """
    A supply chain simulation game based on the beer game by Chaharsooghi (2008).

    Attributes:
    - stockpoints_echelon: list of integers representing the number of nodes per echelon, including suppliers and customers
    - no_suppliers: integer representing the number of suppliers
    - no_customers: integer representing the number of customers
    - no_stockpoints: integer representing the number of stockpoints
    - no_nodes: integer representing the total number of nodes
    - no_echelons: integer representing the total number of echelons, including supplier and customer
    - connections: numpy array representing the connections between every stockpoint
    - unsatisfied_demand: string representing what happens with unsatisfied demand, can be either 'backorders' or 'lost_sales'
    - initial_inventory: numpy array representing the initial inventory per stockpoint
    - holding_costs: numpy array representing the holding costs per stockpoint
    - bo_costs: numpy array representing the backorder costs per stockpoint
    - demand_dist: string representing the demand distribution, can be either 'poisson' or 'uniform'
    - demand_lb: integer representing the lower bound of the demand distribution
    - demand_ub: integer representing the upper bound of the demand distribution
    - leadtime_dist: string representing the leadtime distribution, can only be 'uniform'
    - leadtime_lb: integer representing the lower bound of the leadtime distribution
    - leadtime_ub: integer representing the upper bound of the leadtime distribution
    - order_policy: string representing the predetermined order policy, can be either 'X' or 'X+Y' or 'BaseStock'
    - horizon: integer representing the simulation horizon
    - divide: integer representing the number of time steps per day
    - warmup: integer representing the number of warm-up days
    - fix: boolean representing whether the initial inventory is fixed
    - action_low: numpy array representing the lower bound of the actions
    - action_high: numpy array representing the upper bound of the actions
    - state_scale_low: integer representing the lower bound of the state scaling factor
    - state_scale_high: integer representing the upper bound of the state scaling factor
    - action_min: numpy array representing the minimum action values
    - action_max: numpy array representing the maximum action values
    - state_low: numpy array representing the lower bound of the state variables
    - state_high: numpy array representing the upper bound of the state variables
    """

    def __init__(self):
        # The first element is the number of suppliers
        # The last element is the number of customers
        self.stockpoints_echelon   = [1, 1, 1, 1, 1, 1]
        self.no_suppliers          = self.stockpoints_echelon[0]
        self.no_customers          = self.stockpoints_echelon[-1]
        self.no_stockpoints        = np.sum(self.stockpoints_echelon) - self.no_suppliers - self.no_customers
        self.no_nodes              = np.sum(self.stockpoints_echelon)
        self.no_echelons           = len(self.stockpoints_echelon)
        self.connections           = np.array([
                                      [0, 1, 0, 0, 0, 0],
                                      [0, 0, 1, 0, 0, 0],
                                      [0, 0, 0, 1, 0, 0],
                                      [0, 0, 0, 0, 1, 0],
                                      [0, 0, 0, 0, 0, 1],
                                      [0, 0, 0, 0, 0, 0]
                                      ])
        self.unsatisfied_demand    = 'backorders'                                             
        self.initial_inventory     = [100000, 12, 12, 12, 12, 0] 
        self.holding_costs         = [0, 1, 1, 1, 1, 0]                
        self.bo_costs              = [2, 2, 2, 2, 2, 2]                
        # Demand distribution, can be either 'poisson' or 'uniform'
        self.demand_dist           = 'uniform'
        self.demand_lb             = 0
        self.demand_ub             = 15
        # Leadtime distribution, can only be 'uniform'
        self.leadtime_dist         = 'uniform'
        self.leadtime_lb           = 0
        self.leadtime_ub           = 4                                 
        # Predetermined order policy, can be either 'X' or 'X+Y' or 'BaseStock' 
        self.order_policy          = 'X'
        self.horizon               = 35
        self.divide                = 1000
        self.warmup                = 1
        self.fix                   = False
        self.action_low            = np.array([-1, -1, -1, -1])
        self.action_high           = np.array([1, 1, 1, 1])
        self.state_scale_low                 = -1
        self.state_scale_high                = 1
        self.action_min            = np.array([0,0,0,0])
        self.action_max            = np.array([30,30,30,30])
        self.state_low             = np.zeros([50])
        self.state_high            = np.array([4000,4000,
                                               1000,1000,1000,1000,
                                               1000,1000,1000,1000,
                                               30,30,30,30,
                                               150,150,150,150,
                                               150,150,150,150,
                                               150,150,150,150,
                                               150,150,150,150,
                                               150,150,150,150,
                                               30,30,30,30,
                                               30,30,30,30,
                                               30,30,30,30,
                                               30,30,30,30])

class Divergent:
    """ Based on the paper of Kunnumkal and Topaloglu (2011)"""

    def __init__(self):
        # The first element is the number of suppliers
        # The last element is the number of customers
        self.stockpoints_echelon             = [1, 1, 3, 3]
        self.no_suppliers = self.stockpoints_echelon[0]
        self.no_customers = self.stockpoints_echelon[-1]
        self.no_stockpoints = sum(self.stockpoints_echelon) - \
            self.no_suppliers - self.no_customers
        self.no_nodes = sum(self.stockpoints_echelon)
        self.no_echelons = len(self.stockpoints_echelon)
        self.connections                = np.array([
                                            [0, 1, 0, 0, 0, 0, 0, 0],
                                            [0, 0, 1, 1, 1, 0, 0, 0],
                                            [0, 0, 0, 0, 0, 1, 0, 0],
                                            [0, 0, 0, 0, 0, 0, 1, 0],
                                            [0, 0, 0, 0, 0, 0, 0, 1],
                                            [0, 0, 0, 0, 0, 0, 0, 0],
                                            [0, 0, 0, 0, 0, 0, 0, 0],
                                            [0, 0, 0, 0, 0, 0, 0, 0]
                                            ])
        # Determines what happens with unsatisfied demand, can be either 'backorders' or 'lost_sales'
        self.unsatisfied_demand              = 'backorders'
        self.initial_inventory               = [1000000, 0, 0, 0, 0, 0, 0, 0]   
        self.holding_costs                   = [0, 0.6, 1, 1, 1, 0, 0, 0]      
        self.bo_costs                        = [0, 0, 19, 19, 19, 0, 0, 0]      
        # Demand distribution, can be either 'poisson' or 'uniform' 
        self.demand_dist                     = 'poisson'                       
        self.demand_lb                       = 5                                 
        self.demand_ub                       = 15                                
        # Leadtime distribution, can only be 'uniform'
        self.leadtime_dist                   = 'uniform'                         
        # Lower bound of the leadtime distribution
        self.leadtime_lb                     = 1  
        # Upper bound of the leadtime distribution                               
        self.leadtime_ub                     = 1                                 
        # Predetermined order policy, can be either 'X','X+Y' or 'BaseStock'
        self.order_policy                    = 'X'                           
        self.action_low                      = np.array([-1,-1,-1,-1])
        self.action_high                     = np.array([1,1,1,1])
        self.state_scale_low                 = -1
        self.state_scale_high                = 1
        self.action_min                      = np.array([0,0,0,0])
        self.action_max                      = np.array([300,75,75,75])
        self.state_low                       = np.zeros([13])
        self.state_high                      = np.array([1000, 450,        # Total inventory and total backorders
                                                         250,250,250,250,  # Inventory per stockpoint
                                                         150,150,150,      # Backorders per stockpoint
                                                         150,150,150,150]) # In transit per stockpoint
        self.horizon                         = 75
        self.warmup                          = 25
        self.divide                          = 1000

class General:
    """ Based on the case of the CardBoard Company """

    def __init__(self):
        # Supply chain variables
        # Number of nodes per echelon, including suppliers and customers
        # The first element is the number of suppliers
        # The last element is the number of customers
        self.stockpoints_echelon             = [4, 4, 5, 5]
        # Number of suppliers
        self.no_suppliers = self.stockpoints_echelon[0]
        # Number of customers
        self.no_customers = self.stockpoints_echelon[-1]
        # Number of stockpoints
        self.no_stockpoints = sum(self.stockpoints_echelon) - \
            self.no_suppliers - self.no_customers

        # Total number of nodes
        self.no_nodes = sum(self.stockpoints_echelon)
        # Total number of echelons, including supplier and customer
        self.no_echelons = len(self.stockpoints_echelon)
        # Connections between every stockpoint
        self.relative_rationing = False
        if self.relative_rationing:
            self.connections = np.array([
                                    #0  1  2  3  4  5  6  7    8    9    10   11   12 13 14 15 16 17
                                    [0, 0, 0, 0, 1, 0, 0, 0,   0,   0,    0,   0,   0, 0, 0, 0, 0, 0],   # 0
                                    [0, 0, 0, 0, 0, 1, 0, 0,   0,   0,    0,   0,   0, 0, 0, 0, 0, 0],   # 1
                                    [0, 0, 0, 0, 0, 0, 1, 0,   0,   0,    0,   0,   0, 0, 0, 0, 0, 0],   # 2
                                    [0, 0, 0, 0, 0, 0, 0, 1,   0,   0,    0,   0,   0, 0, 0, 0, 0, 0],   # 3
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0.6, 0.5, 0.15,   0,   0, 0, 0, 0, 0, 0],   # 4
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0.3, 0.4, 0.80, 0.1,   0, 0, 0, 0, 0, 0],   # 5
                                    [0, 0, 0, 0, 0, 0, 0, 0,   0,   0,    0, 0.8, 0.7, 0, 0, 0, 0, 0],   # 6
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0.1, 0.1, 0.05, 0.1, 0.3, 0, 0, 0, 0, 0],   # 7
                                    [0, 0, 0, 0, 0, 0, 0, 0,   0,   0,    0,   0,   0, 1, 0, 0, 0, 0],   # 8
                                    [0, 0, 0, 0, 0, 0, 0, 0,   0,   0,    0,   0,   0, 0, 1, 0, 0, 0],   # 9
                                    [0, 0, 0, 0, 0, 0, 0, 0,   0,   0,    0,   0,   0, 0, 0, 1, 0, 0],   # 10
                                    [0, 0, 0, 0, 0, 0, 0, 0,   0,   0,    0,   0,   0, 0, 0, 0, 1, 0],   # 11
                                    [0, 0, 0, 0, 0, 0, 0, 0,   0,   0,    0,   0,   0, 0, 0, 0, 0, 1],   # 12
                                    [0, 0, 0, 0, 0, 0, 0, 0,   0,   0,    0,   0,   0, 0, 0, 0, 0, 0],   # 13
                                    [0, 0, 0, 0, 0, 0, 0, 0,   0,   0,    0,   0,   0, 0, 0, 0, 0, 0],   # 14
                                    [0, 0, 0, 0, 0, 0, 0, 0,   0,   0,    0,   0,   0, 0, 0, 0, 0, 0],   # 15
                                    [0, 0, 0, 0, 0, 0, 0, 0,   0,   0,    0,   0,   0, 0, 0, 0, 0, 0],   # 16
                                    [0, 0, 0, 0, 0, 0, 0, 0,   0,   0,    0,   0,   0, 0, 0, 0, 0, 0]    # 17
                                    ])
            self.action_low                      = np.array([-1,-1,-1,-1,-1,-1,-1,-1,-1])
            self.action_high                     = np.array([1,1,1,1,1,1,1,1,1])
            self.action_min                      = np.zeros(9)
            self.action_max                      = np.array([150, 150, 150, 150, 75, 75, 75, 75, 75])
        else:
            self.connections = np.array([
                                #row means from who it is 
                                #column means to whom it is connected
                                #0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17
                                [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],   # 0    supplier
                                [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],   # 1    supplier
                                [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],   # 2    supplier
                                [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],   # 3    supplier
                                [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],   # 4    stockpoint
                                [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],   # 5    stockpoint
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],   # 6    stockpoint
                                [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],   # 7    stockpoint
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],   # 8    stockpoint
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],   # 9    stockpoint
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],   # 10   stockpoint
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],   # 11   stockpoint
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],   # 12   stockpoint
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],   # 13   customer
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],   # 14   customer
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],   # 15   customer
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],   # 16   customer
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]    # 17   customer
                                ])
            self.action_low                      = np.array([-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1])
            self.action_high                     = np.array([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1])
            self.action_min                      = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
            self.action_max                      = np.array([150, 150, 150, 150, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75])
        # Determines what happens with unsatisfied demand, can be either 'backorders' or 'lost_sales'
        self.unsatisfied_demand              = 'backorders'  
        # Initial inventory per stockpoint                    
        self.initial_inventory               = [1000000, 1000000, 1000000, 1000000, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        # Holding costs per stockpoint
        self.holding_costs                   = [0, 0, 0, 0, 0.6, 0.6, 0.6, 0.6, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
        # Backorder costs per stockpoint
        self.bo_costs                        = [0, 0, 0, 0, 0, 0, 0, 0, 19, 19, 19, 19, 19, 0, 0, 0, 0, 0]       
        # Demand distribution, can be either 'poisson' or 'uniform'
        self.demand_dist                     = 'poisson'              
        # Lower bound of the demand distribution           
        self.demand_lb                       = 15          
        # Upper bound of the demand distribution                       
        self.demand_ub                       = 15                                
        # Leadtime distribution, can only be 'uniform'
        self.leadtime_dist                   = 'uniform'                         
        # Lower bound of the leadtime distribution
        self.leadtime_lb                     = 1                                 
        # Upper bound of the leadtime distribution
        self.leadtime_ub                     = 1                                
        # Predetermined order policy, can be either 'X' or 'X+Y' or 'BaseStock'
        self.order_policy                    = 'X'                               
        self.horizon                         = 100
        self.warmup                          = 50
        self.divide                          = 1000
        self.state_scale_low                 = 0
        self.state_scale_high                = 1
        self.state_low                       = np.zeros(48)

        self.state_inventory                = 500
        self.state_backorders_warehouses    = 500
        self.state_backorders_retailers     = 250
        self.state_in_transit               = 150


        self.state_high                      = np.array([4500, 8250,                                  # Total inventory and backorders
                                                        500, 500, 500, 500, 500, 500, 500, 500, 500,  # Inventory per stockpoint
                                                        500, 500, 500,                                # Backorders for stockpoint 4
                                                        500, 500, 500, 500,                           # Backorders for stockpoint 5
                                                        500, 500,                                     # Backorders for stockpoint 6
                                                        500, 500, 500, 500, 500,                      # Backorders for stockpoint 7
                                                        250, 250, 250, 250, 250,                      # Backorders for stockpoints 8-12
                                                        150, 150, 150, 150,               # In Transit for stockpoints 4-7
                                                        150, 150, 150,                       # In Transit for stockpoint 8
                                                        150, 150, 150,                       # In Transit for stockpoint 9
                                                        150, 150, 150,                       # In Transit for stockpoint 10
                                                        150, 150, 150,                       # In Transit for stockpoint 11
                                                        150, 150])                          # In Transit for stockpoint 12