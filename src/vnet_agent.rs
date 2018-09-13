use agent::*;
use eventgen::EType;
use gridfuncs::{
    afterstates, argpmax1, get_eligible_chs, get_inuse_chs, incremental_freps, Frep, FrepO, FrepsO,
    CHANNELS, COLS, ROWS,
};
use ndarray::Data;
use ndarray::{Array, Array1, Array2, ArrayBase, ArrayView2, Axis, Dimension};
use std::f32::MIN as fmin;
use std::ops::AddAssign;
use std::ops::SubAssign;

const WDIM: usize = ROWS * COLS * (CHANNELS + 1);

pub trait Net {
    fn forward<S: Data<Elem = f32>, D: Dimension>(
        &mut self,
        freps: &ArrayBase<S, D>,
    ) -> Array1<f32>;

    fn backward<S: Data<Elem = f32>>(
        &mut self,
        frep: &Frep<S>,
        reward: f32,
        avg_reward: f32,
        next_frep: &Frep<S>,
    ) -> f32;
}

pub struct VNet {
    alpha: f32,
    alpha_grad: f32,
    grad_corr: Array2<f32>, // 'w_t': gradient correction weight
    weights: Array2<f32>,   // 'theta_t': neural network weights
}

impl VNet {
    fn new(alpha: f32, alpha_grad: f32) -> Self {
        VNet {
            alpha,
            alpha_grad,
            grad_corr: Array::zeros((WDIM, 1)),
            weights: Array::zeros((WDIM, 1)),
        }
    }
}

impl Net for VNet {
    /// Forward pass. Calculate the state value of one (3D array) or multiple (4D) freps.
    fn forward<S, D>(&mut self, freps: &ArrayBase<S, D>) -> Array1<f32>
    where
        S: Data<Elem = f32>,
        D: Dimension,
    {
        let n_freps = if freps.ndim() == 3 {
            1
        } else {
            freps.len_of(Axis(0))
        };
        // A linear neural network is a simple inner vector product (dot product)
        // between the input and the network weights
        freps
            .view()
            .into_shape((n_freps, WDIM))
            .expect("Freps flatten fail")
            .dot(&self.weights)
            .into_shape(n_freps)
            .expect("State val flatten fail")
    }

    /// Backward pass.
    fn backward<S: Data<Elem = f32>>(
        &mut self,
        frep: &Frep<S>,
        reward: f32,
        avg_reward: f32,
        next_frep: &Frep<S>,
    ) -> f32 {
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
        let c = -2.0 * self.alpha;
        let grads: Array2<f32> =
            (c * td_err) * inp_cv.to_owned() + c * avg_reward - (c * dot) * next_inp_cv.to_owned();
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
    /// Return the state value of each possible afterstate, along with the corresponding
    /// feature representations.
    /// Performs hand-off look-ahead (HLA) for hand-off departures.
    fn get_qvals(&mut self, state: &mut State, chs: &Vec<usize>) -> (Array1<f32>, FrepsO) {
        match state.event.to_cell {
            Some(ref to_cell) => {
                // HLA. This event is is known to be a hand-off departure and the next
                // event is known to be a hand-off arrival.
                let mut end_astates =
                    afterstates(&state.grid, &state.event.cell, &state.event.etype, chs);
                let freps = incremental_freps(
                    &mut state.grid,
                    &state.frep,
                    &state.event.cell,
                    &state.event.etype,
                    chs,
                );
                let mut n_tot = 0;
                let ha_chs: Vec<Vec<usize>> = end_astates
                    .outer_iter()
                    .map(|end_astate| {
                        let echs = get_eligible_chs(&end_astate, &to_cell);
                        n_tot += echs.len();
                        echs
                    })
                    .collect();
                debug_assert_eq!(chs.len(), ha_chs.len());
                debug_assert_eq!(ha_chs.len(), end_astates.len_of(Axis(0)));
                debug_assert_eq!(end_astates.len_of(Axis(0)), freps.len_of(Axis(0)));
                // Shold assert lengths here
                let qvals = if n_tot > 0 {
                    let mut qvals = Array1::zeros(chs.len());
                    for (i, mut end_astate, frep, iha_chs) in izip!(
                        0..,
                        end_astates.outer_iter_mut(),
                        freps.outer_iter(),
                        ha_chs.into_iter()
                    ) {
                        if !iha_chs.is_empty() {
                            let ha_freps = incremental_freps(
                                &mut end_astate,
                                &frep,
                                &to_cell,
                                &EType::HOFF,
                                &iha_chs,
                            );
                            let hla_qvals = self.net.forward(&ha_freps);
                            qvals[[i]] = hla_qvals
                                .fold(fmin, |max, &elem| if elem > max { elem } else { max });
                        }
                    }
                    qvals
                } else {
                    self.net.forward(&freps)
                };
                (qvals, freps)
            }
            _ => {
                let freps = incremental_freps(
                    &mut state.grid,
                    &state.frep,
                    &state.event.cell,
                    &state.event.etype,
                    chs,
                );
                (self.net.forward(&freps), freps)
            }
        }
    }
}

impl Agent for AAVNet<VNet> {
    fn new(alpha: f32, alpha_avg: f32, alpha_grad: f32) -> AAVNet<VNet> {
        AAVNet {
            alpha_avg,
            net: VNet::new(alpha, alpha_grad),
            avg_reward: 0.0,
        }
    }

    /// Select an action and return the Frep
    /// which would result from executing that action.
    fn get_action(&mut self, state: &mut State) -> (Action, FrepO) {
        let chs = match state.event.etype {
            EType::END => get_inuse_chs(&state.grid, &state.event.cell),
            _ => get_eligible_chs(&state.grid, &state.event.cell),
        };
        debug!("Available actions for {:?}: {:?}", state.event, chs);
        if chs.is_empty() {
            assert_ne!(
                state.event.etype,
                EType::END,
                "No channels in use on end event!"
            );
            return (None, state.frep.clone());
        }
        let (qvals, freps) = self.get_qvals(state, &chs);
        // Strictly greedy action selection; no exploration is performed.
        let (idx, _qval) = argpmax1(&qvals).unwrap();
        debug!("qvals: {:?}, idx: {:?}, ch: {}", qvals, idx, chs[idx]);
        (Some(chs[idx]), freps.slice_move(s![idx, .., .., ..]))
    }

    fn update(&mut self, state: &State, _action: Action, reward: i32, next_state: &State) {
        // Knowing the action is not relevant for updating state value nets when
        // both the state and next state are given.
        let err = self.net.backward(
            &state.frep,
            reward as f32,
            self.avg_reward,
            &next_state.frep,
        );
        assert!(
            !err.is_nan(),
            "NaN loss on backprop. Current avg. reward: {}",
            self.avg_reward
        );
        self.avg_reward += self.alpha_avg * err;
    }
}
