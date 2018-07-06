# Reinforcement Learning in Rust
This project implements a RL agent for doing Dynamic Channel Allocation in a 
simulated mobile caller environment.

It is a near-complete Rust port of the best performing agent (AA-VNet) 
from [https://github.com/tsoernes/dca]. This agent utilizes a linear neural network
as state value function approximator which is updated using a newly proposed variant of 
TDC gradients (Sutton et al. 2009: Fast gradient-descent methods for temporal-difference learning with linear function approximation.) for average reward Markov Decision Processes.
