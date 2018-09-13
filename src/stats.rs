use chrono::Local;

#[derive(Default)]
pub struct Stats {
    start_time: i64, // Start time (wall-clock)
    i: i32,          // Current iteration

    // Number of call arrivals this log_iter period (not including hand-offs)
    n_curr_arrivals_new: i32,
    // Number of call arrivals (not including hand-offs)
    n_arrivals_new: i32,
    // Number of accepted call arrivals (not including hand-offs)
    n_accepted_new: i32,
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

    // Block prob during each log iter period
    block_probs: Vec<f64>,
    // For each log_iter,
    // cumulative new/hand-off/total call blocking probability thus far
    cum_block_probs_new: Vec<f64>,
    cum_block_probs_hoff: Vec<f64>,
    cum_block_probs_tot: Vec<f64>,
}

impl Stats {
    pub fn new() -> Self {
        Stats {
            start_time: Local::now().timestamp(),
            ..Default::default()
        }
    }
    pub fn event_arrival_new(&mut self) {
        self.n_curr_arrivals_new += 1;
        self.n_arrivals_new += 1
    }

    pub fn event_arrival_hoff(&mut self) {
        self.n_arrivals_hoff += 1
    }

    pub fn event_accept_new(&mut self) {
        self.n_accepted_new += 1
    }

    pub fn event_reject_new(&mut self) {
        self.n_rejected_new += 1;
        self.n_curr_rejected_new += 1
    }

    pub fn event_reject_hoff(&mut self) {
        self.n_rejected_hoff += 1;
    }

    pub fn event_end(&mut self) {
        self.n_ended += 1;
    }

    fn cums(&mut self) -> (f64, f64, f64) {
        let cum_block_prob_new = self.n_rejected_new as f64 / (self.n_arrivals_new as f64 + 1.0);
        let cum_block_prob_hoff = self.n_rejected_hoff as f64 / (self.n_arrivals_hoff as f64 + 1.0);
        let cum_block_prob_tot = (self.n_rejected_new + self.n_rejected_hoff) as f64
            / (self.n_arrivals_new + self.n_arrivals_hoff + 1) as f64;
        (cum_block_prob_new, cum_block_prob_hoff, cum_block_prob_tot)
    }

    pub fn report_log_iter(&mut self, i: i32) {
        let (cum_block_prob_new, cum_block_prob_hoff, cum_block_prob_tot) = self.cums();
        self.cum_block_probs_new.push(cum_block_prob_new);
        self.cum_block_probs_hoff.push(cum_block_prob_hoff);
        self.cum_block_probs_tot.push(cum_block_prob_tot);

        let block_prob = self.n_curr_rejected_new as f64 / (self.n_curr_arrivals_new as f64 + 1.0);
        self.block_probs.push(block_prob);
        println!(
            "\nBlocking probability events {}-{}: {:.4}, cumulative {:.4}",
            self.i, i, block_prob, cum_block_prob_new
        );
        self.n_curr_rejected_new = 0;
        self.n_curr_arrivals_new = 0;
        self.i = i;
    }

    /// t: Simulation time
    /// n_in_progress: Number of calls currently in progress at simulation end
    pub fn report_end(&mut self, t: f64, n_in_progress: usize) {
        // Count how many calls _should_ currently be in progress, based on the number
        // of reported incoming and terminated calls
        let delta = self.n_arrivals_new + self.n_arrivals_hoff
            - self.n_rejected_new
            - self.n_rejected_hoff
            - self.n_ended;
        assert_eq!(delta, n_in_progress as i32);
        let dt = (Local::now().timestamp() - self.start_time) as f64;
        let m = dt / 60.0;
        let h = dt - m * 60.0;
        let rate = self.i as f64 / dt;
        println!(
            "\nSimulation duration: {:.2} sim hours, {}m{}s real, {} events at {:.0} events/second",
            t / 60.0,
            m,
            h,
            self.i,
            rate
        );

        debug!(
            "n_curr_arrivals_new: {},
             n_arrivals_new: {},
             n_accepted_new: {},
             n_arrivals_hoff: {},
             n_ended: {},
             n_curr_rejected_new: {},
             n_rejected_new: {},
             n_rejected_hoff: {}",
            self.n_curr_arrivals_new,
            self.n_arrivals_new,
            self.n_accepted_new,
            self.n_arrivals_hoff,
            self.n_ended,
            self.n_curr_rejected_new,
            self.n_rejected_new,
            self.n_rejected_hoff
        );

        let (cum_block_prob_new, cum_block_prob_hoff, cum_block_prob_tot) = self.cums();
        println!(
            "Blocking probability: {:.4} for new calls",
            cum_block_prob_new
        );
        if self.n_rejected_hoff > 0 {
            print!(
                ", {:.4} for hand-offs, {:.4} total",
                cum_block_prob_hoff, cum_block_prob_tot
            );
        }
    }
}
