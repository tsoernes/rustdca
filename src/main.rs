mod agent;
mod environment;
mod eventgen;
mod gridfuncs;
mod stats;
mod vnet_agent;

extern crate ctrlc;
#[macro_use]
extern crate ndarray;
#[macro_use]
extern crate lazy_static;
extern crate ordered_float;
extern crate rand;
extern crate revord;
#[macro_use]
extern crate structopt;
extern crate chrono;

use structopt::StructOpt;

// arg_enum! {
//     #[derive(Debug)]
//     enum Baz {
//         Foo,
//         Bar,
//         FooBar
//     }
// }
// #[structopt(raw(possible_values = "&Baz::variants()", case_insensitive = "true"))]
// i: Baz,

#[derive(StructOpt, Debug)]
#[structopt(name = "DCA")]
pub struct Opt {
    /// Call duration, in minutes
    #[structopt(short = "d", long = "call_dur", default_value = "3")]
    call_dur: f32,

    /// Call rate, in calls per hour
    #[structopt(short = "r", long = "call_rate", default_value = "200")]
    call_rate_ph: f32,

    /// Hand-off probability
    #[structopt(short = "phoff", long = "p_handoff", default_value = "0.0")]
    p_hoff: f32,

    /// Simulation duration
    #[structopt(short = "i", long = "n_events", default_value = "470000")]
    n_events: i32,

    /// Simulation duration
    #[structopt(long = "log_iter", default_value = "25000")]
    log_iter: i32,

    /// Learning rate 2.52e-6
    #[structopt(short = "lr", long = "alpha", default_value = "2.52e-6")]
    alpha: f32,

    /// Learning rate for average reward 4.75e-5
    #[structopt(short = "alr", long = "alpha_avg", default_value = "0.02")]
    alpha_avg: f32,

    /// Learning rate for TDC gradient corrections 5e-6
    #[structopt(short = "glr", long = "alpha_grad", default_value = "0.02")]
    alpha_grad: f32,

    /// Verify channel reuse constraint each iteration
    #[structopt(long = "verify_grid")]
    verify_grid: bool,
}

fn main() {
    let opt = Opt::from_args();
    println!("{:?}", opt);
}
