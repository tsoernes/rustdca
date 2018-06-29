use eventgen::Event;
use std::cmp::max;

struct Stats {
    i: i32, // Current iteration
    t: f32, // Current time

    // Number of call arrivals this log_iter period (not including hand-offs)
    n_curr_arrivals_new: i32,
    // Number of call arrivals (not including hand-offs)
    n_arrivals_new: i32,
    // Number of hand-offs arrival
    n_arrivals_hoff: i32,
    // Number of ended calls (including handed-off calls)
    n_ended: i32,
    // Number of rejected calls this log_iter period (not including hand-offs)
    n_curr_rejected_new: i32,
    // Number of rejected calls (not including hand-offs)
    n_rejected_new: i32,
    // Number of rejected hand-offs
    n_rejected_hoff: i32,

    //
    block_probs: Vec<f32>,
    // For each log_iter,
    // cumulative new/hand-off/total call blocking probability thus far
    cum_block_probs_new: Vec<f32>,
    cum_block_probs_hoff: Vec<f32>,
    cum_block_probs_tot: Vec<f32>,
}

impl Stats {
    pub fn event_arrival_new(&mut self) {
        self.n_curr_arrivals_new += 1;
        self.n_arrivals_new += 1
    }

    pub fn event_arrival_hoff(&mut self) {
        self.n_arrivals_hoff += 1
    }

    pub fn reject_new(&mut self) {
        self.n_rejected_new += 1;
        self.n_curr_rejected_new += 1
    }

    pub fn reject_hoff(&mut self) {
        self.n_rejected_hoff += 1;
    }

    pub fn log_iter(&mut self, i: i32, event: Event) {
        self.i = i;
        self.t = event.time;

        let cum_block_prob_new: f32 =
            self.n_arrivals_new as f32 / max(self.n_arrivals_new as f32, 1.0);
        self.cum_block_probs_new.append(cum_block_prob_new);
    }
}
