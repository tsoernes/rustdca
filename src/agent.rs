use super::Opt;
use environment::Env;
use eventgen::{EType, Event};
use gridfuncs::{Cell, Grid, CHANNELS, COLS, ROWS};

pub struct Action {
    cell: Cell,
    ch: Option<usize>,
    etype: EType,
}

pub trait Agent {
    fn get_init_action(&mut self, next_event: &Event) -> Option<usize>;
    fn get_action(&mut self, next_event: &Event, grid: &Grid, action: &Action) -> Option<usize>;
    fn optimal_ch(etype: EType, cell: &Cell, grid: &Grid) -> Option<usize>;
}

pub struct AAVNet {
    a: usize,
}

impl AAVNet {
    fn new() -> AAVNet {
        AAVNet { a: 0 }
    }
}

impl Agent for AAVNet {
    fn get_init_action(&mut self, next_event: &Event) -> Option<usize> {
        Some(1)
    }
    fn get_action(&mut self, next_event: &Event, grid: &Grid, action: &Action) -> Option<usize> {
        Some(1)
    }
    fn optimal_ch(etype: EType, cell: &Cell, grid: &Grid) -> Option<usize> {
        Some(1)
    }
}

// fn simulate(opt: Opt) -> (f64, f64) {
fn simulate(opt: Opt) {
    let (mut env, event) = Env::new(opt.p_hoff, opt.verify_grid);
    let t: f32 = 0.0; // Simulation time
    let mut agent = AAVNet::new();
    let counter = ctrlc::Counter::new(ctrlc::SignalType::Ctrlc).unwrap();

    // use std::thread;
    // use std::time;
    if counter.get().unwrap() == 1 {
        println!("Got CTRL-C");
    }

    let mut ch = agent.get_init_action(&event);
    for i in 0..opt.n_events {
        let old_grid = env.grid.clone();
        let old_action = Action {
            cell: event.cell.clone(),
            ch: ch,

            etype: event.etype.clone(),
        };
        let (reward, event) = env.step(event.copy(), ch);
        ch = agent.get_action(&event, &env.grid, &old_action);
        if i > 0 && i % opt.log_iter == 0 {
            env.stats.report_log_iter(i);
        }
    }
}
