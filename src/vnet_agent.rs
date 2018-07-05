use agent::*;
use eventgen::EType;
use gridfuncs::{
    argpmax1, get_eligible_chs, get_inuse_chs, incremental_freps, Frep, CHANNELS, COLS, ROWS,
};
use ndarray::{Array, Array1, Array2, ArrayView2, Axis, Dimension};
use std::ops::AddAssign;
use std::ops::SubAssign;

const WDIM: usize = ROWS * COLS * (CHANNELS + 1);

pub trait Net {
    fn forward<D: Dimension>(&mut self, freps: &Array<f32, D>) -> Array1<f32>;
    fn backward(&mut self, frep: &Frep, reward: f32, avg_reward: f32, next_frep: &Frep) -> f32;
}

pub struct VNet {
    alpha: f32,
    alpha_grad: f32,
    grad_corr: Array2<f32>, // w_t
    weights: Array2<f32>,   // theta_t
}

impl VNet {
    fn new(alpha: f32, alpha_grad: f32) -> Self {
        let w1: Array2<f32> = Array::zeros((WDIM, 1));
        let w2: Array2<f32> = Array::zeros((WDIM, 1));
        VNet {
            alpha: alpha,
            alpha_grad: alpha_grad,
            grad_corr: w1,
            weights: w2,
        }
    }
}

impl Net for VNet {
    /// Forward pass. Calculate the state value of 1 Frep or many Freps
    fn forward<D: Dimension>(&mut self, freps: &Array<f32, D>) -> Array1<f32> {
        let n_freps = if freps.ndim() == 3 {
            1
        } else {
            freps.len_of(Axis(0))
        };
        // println!("Reshape {:?} -> {:?}", freps.shape(), (n_freps, WDIM));
        let inp_rv: ArrayView2<f32> = freps
            .view()
            .into_shape((n_freps, WDIM))
            .expect("Frep reshape");
        inp_rv
            .dot(&self.weights)
            .into_shape(n_freps)
            .expect("Frep reshape2")
    }

    /// Backward pass.
    fn backward(&mut self, frep: &Frep, reward: f32, avg_reward: f32, next_frep: &Frep) -> f32 {
        let value = self.forward(frep);
        assert_eq!(value.shape(), &[1]);
        let value = value[[0]];
        let next_value = self.forward(next_frep)[[0]];
        let td_err = reward - avg_reward + next_value - value;
        let inp_rv: ArrayView2<f32> = frep.view().into_shape((1, WDIM)).expect("Frep reshape3");
        let inp_cv = inp_rv.t();
        let next_inp_cv: ArrayView2<f32> = next_frep
            .view()
            .into_shape((WDIM, 1))
            .expect("Frep reshape4");
        let dot = inp_rv.dot(&self.grad_corr);
        assert_eq!(dot.shape(), &[1, 1]);
        let dot = dot[[0, 0]];
        let c = 2.0 * self.alpha;
        let grads: Array2<f32> =
            (c * td_err) * inp_cv.to_owned() - c * avg_reward + (c * dot) * next_inp_cv.to_owned();
        assert_eq!(grads.shape(), self.weights.shape());
        self.weights.sub_assign(&grads);
        let upd = (self.alpha_grad * (td_err - dot)) * inp_cv.to_owned();
        assert_eq!(upd.shape(), self.grad_corr.shape());
        self.grad_corr.add_assign(&upd);
        td_err
    }
}

pub struct AAVNet<N: Net> {
    alpha_avg: f32,
    net: N,
    avg_reward: f32,
}

impl<N: Net> AAVNet<N> {
    fn get_qvals(&mut self, state: &mut State, chs: &Vec<usize>) -> Array1<f32> {
        let freps = incremental_freps(
            &mut state.grid,
            &state.frep,
            &state.event.cell,
            &state.event.etype,
            chs,
        );
        self.net.forward(&freps)
    }
}

impl Agent for AAVNet<VNet> {
    fn new(alpha: f32, alpha_avg: f32, alpha_grad: f32) -> AAVNet<VNet> {
        AAVNet {
            alpha_avg: alpha_avg,
            net: VNet::new(alpha, alpha_grad),
            avg_reward: 0.0,
        }
    }

    fn get_action(&mut self, state: &mut State) -> Action {
        let chs = match state.event.etype {
            EType::END => get_inuse_chs(&state.grid, &state.event.cell),
            _ => get_eligible_chs(&state.grid, &state.event.cell),
        };
        debug!("Available actions for {:?}: {:?}", state.event, chs);
        if chs.is_empty() {
            assert_ne!(
                state.event.etype,
                EType::END,
                "No channels in use for end event!"
            );
            return None;
        }
        // TODO HLA
        let qvals = self.get_qvals(state, &chs);
        let (idx, _qval) = argpmax1(&qvals).unwrap();
        debug!("qvals: {:?}, idx: {:?}, ch: {}", qvals, idx, chs[idx]);
        Some(chs[idx])
    }

    fn update(&mut self, state: &State, _action: Action, reward: i32, next_state: &State) {
        let err = self.net.backward(
            &state.frep,
            reward as f32,
            self.avg_reward,
            &next_state.frep,
        );
        assert!(!err.is_nan(), "NaN loss");
        self.avg_reward += self.alpha_avg * err;
    }
}
