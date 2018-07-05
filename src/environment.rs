use super::Opt;
use agent::Action;
use eventgen::*;
use gridfuncs::*;
use ndarray::{Array, Array3};
use rand::{thread_rng, Rng};
use stats::Stats;

pub struct Env {
    p_handoff: f32,
    verify_grid: bool,
    pub grid: Grid,
    pub stats: Stats,
    eventgen: EventGen,
}

impl Env {
    /// Initialize an environment and return the first event to be processed
    pub fn new(opt: &Opt) -> (Env, Event) {
        let grid: Array3<bool> = Array::default((ROWS, COLS, CHANNELS));
        let mut eventgen = EventGen::new(opt);
        for r in 0..ROWS {
            for c in 0..COLS {
                eventgen.event_new(0.0, Cell { row: r, col: c })
            }
        }
        let event = eventgen.pop();
        (
            Env {
                p_handoff: opt.p_hoff,
                verify_grid: opt.verify_grid,
                grid: grid,
                stats: Stats::new(),
                eventgen: eventgen,
            },
            event,
        )
    }

    pub fn step(&mut self, event: Event, action: Action) -> (usize, Event) {
        let (time, cell) = (event.time, event.cell);
        debug!("Time: {}, etype: {}, ch: {:?}", time, event.etype, action);
        match event.etype {
            EType::NEW => {
                self.stats.event_arrival_new();
                self.eventgen.event_new(time, cell);
                match action {
                    Some(ch) => {
                        self.stats.event_accept_new();
                        let p = thread_rng().gen::<f32>();
                        if p < self.p_handoff {
                            self.eventgen.event_hoff_new(time, cell, ch);
                        } else {
                            self.eventgen.event_end(time, cell, ch);
                        }
                    }
                    None => {
                        self.stats.event_reject_new();
                    }
                }
            }
            EType::HOFF => {
                self.stats.event_arrival_hoff();
                match action {
                    Some(ch) => {
                        self.eventgen.event_hoff_end(time, cell, ch);
                    }
                    None => {
                        self.stats.event_reject_hoff();
                    }
                }
            }
            EType::END => {
                self.stats.event_end();
                assert!(action.is_some())
            }
        }
        action.map(|ch| self.execute_action(event, ch));
        if self.verify_grid {
            assert!(validate_reuse_constraint(&self.grid));
        }
        let reward = n_used(&self.grid);
        debug!("Reward: {}", reward);
        (reward, self.eventgen.pop())
    }

    pub fn execute_action(&mut self, event: Event, ch: usize) {
        debug!("Executing action {:?}, {}", event, ch);
        let (r, c) = (event.cell.row, event.cell.col);
        let dbgstr = format!(
            "Unexpected grid state when executing action {:?}, CH: {}",
            event, ch
        );
        match event.etype {
            EType::END => {
                let reass_ch = event.ch.expect("No CH for end event");
                assert!(self.grid[[r, c, reass_ch]], dbgstr);
                if reass_ch != ch {
                    assert!(self.grid[[r, c, ch]], dbgstr);
                    self.eventgen.reassign(event.cell, ch, reass_ch);
                }
                self.grid[[r, c, ch]] = false;
            }
            _ => {
                assert!(!self.grid[[r, c, ch]], dbgstr);
                self.grid[[r, c, ch]] = true;
            }
        }
    }
}
