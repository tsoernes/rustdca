use eventgen::*;
use gridfuncs::*;
use rand::{thread_rng, Rng};

struct Env {
    p_handoff: f32,
    verify_grid: bool,
    grid: Grid,
    stats: usize, // TODO
    eventgen: EventGen,
    event: Event,
}

impl Env {
    pub fn init_calls(&mut self) {
        for r in 0..ROWS {
            for c in 0..COLS {
                self.eventgen.event_new(0.0, Cell { row: r, col: c })
            }
        }
        self.event = self.eventgen.pop()
    }

    pub fn step(&mut self, ch: Option<usize>) -> usize {
        let time = self.event.time;
        let cell = self.event.cell;
        // self.stats.iter(t, )
        match self.event.etype {
            EType::NEW => {
                // self.stats.event_new();
                self.eventgen.event_new(time, cell);
                match ch {
                    Some(ch) => {
                        let p = thread_rng().gen::<f32>();
                        if p < self.p_handoff {
                            self.eventgen.event_hoff_new(
                                time,
                                cell,
                                ch,
                                neighbors(1, cell.row, cell.col, false),
                            );
                        } else {
                            self.eventgen.event_end(time, cell, ch);
                        }
                    }
                    None => {
                        // stats.event_new_reject
                        ()
                    }
                }
            }
            EType::HOFF => {
                // self.stats.event_hoff_new();
                match ch {
                    Some(ch) => {
                        self.eventgen.event_hoff_end(time, cell, ch);
                    }
                    None => {
                        // self.stats.event_hoff_reject();
                        ()
                    }
                }
            }
            EType::END => {
                // self.stats.event_end();
                assert!(ch.is_some())
            }
        }
        ch.map(|ch| self.execute_action(ch));
        if self.verify_grid {
            assert!(validate_reuse_constraint(&self.grid));
        }
        self.event = self.eventgen.pop();
        n_used(&self.grid)
    }

    pub fn execute_action(&mut self, ch: usize) {
        let (r, c) = (self.event.cell.row, self.event.cell.col);
        match self.event.etype {
            EType::END => {
                let reass_ch = self.event.ch.expect("No CH for end event");
                assert!(self.grid[[r, c, reass_ch]]);
                if reass_ch != ch {
                    assert!(self.grid[[r, c, ch]]);
                    self.eventgen.reassign(self.event.cell, ch, reass_ch);
                }
                self.grid[[r, c, ch]] = false
            }
            _ => {
                assert!(!self.grid[[r, c, ch]]);
                self.grid[[r, c, ch]] = true
            }
        }
    }
}
