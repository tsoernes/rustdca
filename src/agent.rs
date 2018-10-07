use super::Opt;
use ctrlc::set_handler;
use environment::Env;
use eventgen::Event;
use gridfuncs::{feature_rep, n_used, FrepO, GridO};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

pub struct State {
    pub grid: GridO,
    pub frep: FrepO,
    pub event: Event,
}

pub type Action = Option<usize>;

pub trait Agent {
    fn new(alpha: f32, alpha_avg: f32, alpha_grad: f32) -> Self;
    fn get_action(&mut self, state: &mut State) -> (Action, FrepO);
    fn update(&mut self, state: &State, action: Action, reward: i32, next_state: &State);
}

/// (x_t, e_t) -> a_t -> r_{t+1} -> (x_{t+1}, e_{t+1})
pub fn simulate<A: Agent>(opt: &Opt) {
    // Create a CTRL-C key handler
    let running = Arc::new(AtomicBool::new(true));
    let r = running.clone();
    set_handler(move || {
        r.store(false, Ordering::SeqCst);
    }).expect("Error setting Ctrl-C handler");

    // Initialize the agent and the environment; get the first
    // call event to handle and the first action response to that event
    let (mut env, event) = Env::new(opt);
    let mut agent: A = A::new(opt.alpha, opt.alpha_avg, opt.alpha_grad);
    let mut state = State {
        grid: env.grid.clone(),
        frep: feature_rep(&env.grid),
        event: event,
    };
    let (mut action, mut next_frep) = agent.get_action(&mut state);
    let mut next_state;
    for i in 0..opt.n_events {
        if !running.load(Ordering::SeqCst) {
            println!("Premature exit");
            break;
        }
        let (reward, next_event) = env.step(state.event.clone(), action);
        next_state = State {
            grid: env.grid.clone(),
            frep: next_frep,
            event: next_event,
        };
        agent.update(&state, action, reward as i32, &next_state);
        let (a, f) = agent.get_action(&mut next_state);
        action = a;
        next_frep = f;
        state = next_state;

        if i > 0 && i % opt.log_iter == 0 {
            env.stats.report_log_iter(i);
        }
    }
    env.stats.report_end(state.event.time, n_used(&env.grid));
}
