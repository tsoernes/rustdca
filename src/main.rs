#[macro_use]
extern crate ndarray;

use ndarray::prelude::*;
use std::ops::AddAssign;


fn hex_distance(r1: i8, c1: i8, r2: i8, c2: i8) -> i8 {
    // Distance from cell (r1, c1) to cell (r2, c2) in a hexagonal grid
    ((r1 - r2).abs() + (r1 + c1 - r2 - c2).abs() + (c1 - c2).abs()) / 2
}

fn generate_neighs() -> (Array<usize, Ix4>, Array<usize, Ix4>,Array<usize, Ix4>,
                        Array<usize, Ix4>, Array<usize, Ix3>) {
    // Return 4 arrays with indecies of neighbors (including self) with distance of (1..5) or less;
    // and a (4 X rows x cols) array with the number of neighbors for each row and column since
    // the number of n-distance neighbors can vary for each cell.
    let rows = 7;
    let cols = 7;
    // Indecies of neighbors with distance of 1 or less
    let mut neighs1 = Array::zeros((rows, cols, 7, 2));
    let mut neighs2 = Array::zeros((rows, cols, 19, 2));
    let mut neighs3 = Array::zeros((rows, cols, 37, 2));
    let mut neighs4 = Array::zeros((rows, cols, 43, 2));
    // For distance 1..5 or less, for each row and col: the number of neighbors.
    let mut n_neighs = Array::zeros((4, rows, cols));
    neighs1.slice_mut(s![0, 0, 0, ..]).assign(&Array::ones(2));
    for r1 in 0..rows {
        for c1 in 0..cols {
            // Want to store index of self first, so that it can be easily excluded
            neighs1.slice_mut(s![r1, c1, 0, ..]).assign(&array![r1, c1]);
            n_neighs.slice_mut(s![.., r1, c1]).add_assign(1);
            for r2 in 0..rows {
                for c2 in 0..cols {
                    let dist = hex_distance(r1 as i8, c1 as i8, r2 as i8, c2 as i8);
                    if (r1, c1) != (r2, c2) && dist <= 4 {
                        let n = n_neighs[[3, r1, c1]];
                        neighs4 .slice_mut(s![r1, c1, n, ..])
                                .assign(&array![r2, c2]);
                        n_neighs[[3, r1, c1]] += 1;
                        if dist <= 3 {
                            let n = n_neighs[[2, r1, c1]];
                            neighs3 .slice_mut(s![r1, c1, n, ..])
                                    .assign(&array![r2, c2]);
                            n_neighs[[2, r1, c1]] += 1;
                            if dist <= 2 {
                                let n = n_neighs[[1, r1, c1]];
                                neighs2 .slice_mut(s![r1, c1, n, ..])
                                        .assign(&array![r2, c2]);
                                n_neighs[[1, r1, c1]] += 1;
                                if dist == 1 {
                                    let n = n_neighs[[0, r1, c1]];
                                    neighs1 .slice_mut(s![r1, c1, n, ..])
                                            .assign(&array![r2, c2]);
                                    n_neighs[[0, r1, c1]] += 1;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    return (neighs1, neighs2, neighs3, neighs4, n_neighs)
}

static neighs: (Array<usize, Ix4>, Array<usize, Ix4>,Array<usize, Ix4>, Array<usize, Ix4>, Array<usize, Ix3>) = generate_neighs();

fn main() {
    // let neighs = generate_neighs();
    println!("{}", neighs.4);
}
