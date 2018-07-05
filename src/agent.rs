use super::Opt;
use ctrlc::set_handler;
use environment::Env;
use eventgen::Event;
use gridfuncs::{feature_rep, n_used, Frep, Grid};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

pub struct State {
    pub grid: Grid,
    pub frep: Frep,
    pub event: Event,
}

pub type Action = Option<usize>;

pub trait Agent {
    fn new(alpha: f32, alpha_avg: f32, alpha_grad: f32) -> Self;
    fn get_action(&mut self, state: &mut State) -> Action;
    fn update(&mut self, state: &State, action: Action, reward: i32, next_state: &State);
}

/// (x_t, e_t) -> a_t -> r_{t+1} -> (x_{t+1}, e_{t+1})
// fn simulate(opt: Opt) -> (f32, f32) {
pub fn simulate<A: Agent>(opt: &Opt) {
    // Create CTRL-C key handler
    let running = Arc::new(AtomicBool::new(true));
    let r = running.clone();
    set_handler(move || {
        r.store(false, Ordering::SeqCst);
    }).expect("Error setting Ctrl-C handler");

    let (mut env, event) = Env::new(opt);
    let mut agent: A = A::new(opt.alpha, opt.alpha_avg, opt.alpha_grad);

    let mut state = State {
        grid: env.grid.clone(),
        frep: feature_rep(&env.grid),
        event: event,
    };
    let mut action = agent.get_action(&mut state);
    let mut next_state;
    for i in 0..opt.n_events {
        if !running.load(Ordering::SeqCst) {
            println!("Premature exit");
            break;
        }
        let (reward, next_event) = env.step(state.event.clone(), action);
        next_state = State {
            grid: env.grid.clone(),
            // TODO incremental freps
            frep: feature_rep(&env.grid),
            event: next_event,
        };
        agent.update(&state, action, reward as i32, &next_state);
        action = agent.get_action(&mut next_state);

        if i > 0 && i % opt.log_iter == 0 {
            env.stats.report_log_iter(i);
        }
        state = next_state;
    }
    env.stats.report_end(state.event.time, n_used(&env.grid));
}
