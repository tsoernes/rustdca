# Dynamic Channel Allocation by Reinforcement Learning
This project implements a RL agent for doing Dynamic Channel Allocation in a 
simulated mobile caller environment.

The implementation is in Rust and uses [ndarray](https://docs.rs/ndarray/) for numerical work.

It is a near-complete Rust port of the best performing agent (AA-VNet) 
from https://github.com/tsoernes/dca. This agent utilizes a linear neural network
as state value function approximator which is updated using a newly proposed variant of 
TDC gradients, originally defined in [Sutton et al. 2009](https://www.ics.uci.edu/~dechter/courses/ics-295/winter-2018/papers/2009-sutton-Fast_gradient-descent.pdf): 
"Fast gradient-descent methods for temporal-difference learning with linear function approximation."

See also the version written in [Haskell](https://github.com/tsoernes/haskelldca),
and [Python](https://github.com/tsoernes/dca).
# How to build
```
cargo build --release
```
# How to run
```
cargo run --release -- --n_events 100_000
```
or, to see the help menu for all command line arguments:
```
cargo run --release -- --help
USAGE:
    rustdca [FLAGS] [OPTIONS]

FLAGS:
    -h, --help           Prints help information
    -V, --version        Prints version information
    -v, --verbose        Log level: '-v' for debug, '-vv' for trace
        --verify_grid    Verify channel reuse constraint each iteration

OPTIONS:
    -l, --alpha <alpha>                    Learning rate for neural network [default: 2.52e-6]
    -a, --alpha_avg <alpha_avg>            Learning rate for average reward [default: 0.06]
    -g, --alpha_grad <alpha_grad>          Learning rate for TDC gradient corrections [default: 5e-6]
        --call_dur <call_dur>              Call duration, in minutes [default: 3]
    -r, --call_rate <call_rate_ph>         Call rate, in calls per hour [default: 200]
        --hoff_call_dur <hoff_call_dur>    Call duration for hand-offs, in minutes [default: 1]
        --log_iter <log_iter>              Show blocking probability every 'log_iter' iterations [default: 5000]
    -i, --n_events <n_events>              Simulation duration [default: 10000]
    -p, --p_handoff <p_hoff>               Hand-off probability [default: 0.0]
```
