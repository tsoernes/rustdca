#[macro_use]
extern crate ndarray;
#[macro_use]
extern crate lazy_static;

use ndarray::prelude::*;
use std::ops::AddAssign;
use std::ops::BitOr;
use std::ops::BitOrAssign;
use std::ops::Not;

const ROWS: usize = 7;
const COLS: usize = 7;
const CHANNELS: usize = 70;

lazy_static! {
    // (NEIGHS1, NEIGHS2, NEIGHS4, N_NEIGHS):
    static ref NEIGHS: (Array<usize, Ix4>, Array<usize, Ix4>,
                        Array<usize, Ix4>, Array<usize, Ix3>) = generate_neighs();
}

pub struct Cell {
    row: usize,
    col: usize,
}

#[derive(PartialEq)]
pub enum CEType {
    NEW,
    END,
    HOFF,
}

struct CEvent {
    ce_type: CEType,
    cell: Cell,
}

type Grid = Array<bool, Ix3>;
type Grids = Array<bool, Ix4>;
type Frep = Array<usize, Ix3>;
type Freps = Array<usize, Ix4>;

/// Distance from cell (r1, c1) to cell (r2, c2) in a hexagonal grid
fn hex_distance(r1: i8, c1: i8, r2: i8, c2: i8) -> i8 {
    ((r1 - r2).abs() + (r1 + c1 - r2 - c2).abs() + (c1 - c2).abs()) / 2
}

/// Return 3 arrays with indecies of neighbors (including self) with distance of 1,2,4 or less;
/// and a (3 X rows x cols) array with the number of neighbors for each row and column since
/// the number of n-distance neighbors can vary for each cell.
fn generate_neighs() -> (
    Array<usize, Ix4>,
    Array<usize, Ix4>,
    Array<usize, Ix4>,
    Array<usize, Ix3>,
) {
    // Indecies of neighbors with distance of 1 or less
    let mut neighs1 = Array::zeros((ROWS, COLS, 7, 2));
    let mut neighs2 = Array::zeros((ROWS, COLS, 19, 2));
    let mut neighs4 = Array::zeros((ROWS, COLS, 43, 2));
    // For each distance above, for each row and col: the number of neighbors.
    let mut n_neighs = Array::zeros((3, ROWS, COLS));
    neighs1.slice_mut(s![0, 0, 0, ..]).assign(&Array::ones(2));
    for r1 in 0..ROWS {
        for c1 in 0..COLS {
            // Want to store index of self first, so that it can be easily excluded
            neighs1.slice_mut(s![r1, c1, 0, ..]).assign(&array![r1, c1]);
            n_neighs.slice_mut(s![.., r1, c1]).add_assign(1);
            for r2 in 0..ROWS {
                for c2 in 0..COLS {
                    let dist = hex_distance(r1 as i8, c1 as i8, r2 as i8, c2 as i8);
                    if (r1, c1) != (r2, c2) && dist <= 4 {
                        let n = n_neighs[[2, r1, c1]];
                        neighs4.slice_mut(s![r1, c1, n, ..]).assign(&array![r2, c2]);
                        n_neighs[[2, r1, c1]] += 1;
                        if dist <= 2 {
                            let n = n_neighs[[1, r1, c1]];
                            neighs2.slice_mut(s![r1, c1, n, ..]).assign(&array![r2, c2]);
                            n_neighs[[1, r1, c1]] += 1;
                            if dist == 1 {
                                let n = n_neighs[[0, r1, c1]];
                                neighs1.slice_mut(s![r1, c1, n, ..]).assign(&array![r2, c2]);
                                n_neighs[[0, r1, c1]] += 1;
                            }
                        }
                    }
                }
            }
        }
    }
    (neighs1, neighs2, neighs4, n_neighs)
}

fn neighbors(
    dist: usize,
    row: usize,
    col: usize,
    include_self: bool,
) -> ArrayView<'static, usize, Ix2> {
    // [[row1, row2, .., rowN], [col1, col2, .., colN]] for N neighbors
    let j = &NEIGHS;
    let allneighs = match dist {
        1 => &j.0,
        2 => &j.1,
        4 => &j.2,
        _ => panic!("Neighbors for distances other than 1, 2 or 4 should never be needed"),
    };
    let start = if include_self { 0 } else { 1 };
    let end = NEIGHS.3[[dist - 1, row, col]];
    allneighs.slice(s![row, col, start..end, ..])
}

/// Alloc. map of channels in use at cell neighbors with distance of 2 or less
fn inuse_neighs(grid: &Grid, cell: &Cell) -> Array<bool, Ix1> {
    let neighs = neighbors(2, cell.row, cell.col, false);
    let mut alloc_map: Array1<bool> = Array::default(CHANNELS);
    for neigh in neighs.outer_iter() {
        alloc_map.bitor_assign(&grid.slice(s![neigh[0], neigh[1], ..]));
    }
    alloc_map
}

/// Channels that are not in use at cell or its neighbors with distance of 2 or less
fn eligible_map(grid: &Grid, cell: &Cell) -> Array<bool, Ix1> {
    inuse_neighs(grid, cell)
        .bitor(&grid.slice(s![cell.row, cell.col, ..]))
        .not()
}

fn nonzero1(array: &Array<bool, Ix1>) -> Vec<usize> {
    array
        .indexed_iter()
        .filter_map(|(index, &item)| if item { Some(index) } else { None })
        .collect()
}

fn get_eligible_chs(grid: &Grid, cell: &Cell) -> Vec<usize> {
    eligible_map(grid, cell)
        .indexed_iter()
        .filter_map(|(index, &item)| if item { Some(index) } else { None })
        .collect()
}

fn afterstates(grid: &Grid, cell: &Cell, ce_type: &CEType, chs: &[usize]) -> Grids {
    let targ_val = match ce_type {
        CEType::END => false,
        _ => true,
    };
    let mut grids: Grids = Array::default((chs.len(), 7, 7, 70));
    for (i, ch) in chs.into_iter().enumerate() {
        grids.slice_mut(s![i, .., .., ..]).assign(&grid);
        grids[[i, cell.row, cell.col, *ch]] = targ_val;
    }
    grids
}

/// Returns false if reuse constraint is violated
pub fn validate_reuse_constraint(grid: &Grid) -> bool {
    for r in 0..ROWS {
        for c in 0..COLS {
            let cell = Cell { row: r, col: c };
            let inuse = inuse_neighs(grid, &cell).bitor(&grid.slice(s![cell.row, cell.col, ..]));
            if inuse.into_iter().any(|&x| x) {
                return false;
            }
        }
    }
    true
}

pub fn feature_rep(grid: &Grid) -> Frep {
    let mut frep = Array::zeros((ROWS, COLS, CHANNELS + 1));
    // Probably inefficient
    let g = grid.mapv(|x: bool| x as usize);
    for r in 0..ROWS {
        for c in 0..COLS {
            let neighs = neighbors(4, r, c, false);
            let mut n_used: Array1<usize> = Array::zeros(CHANNELS);
            for neigh in neighs.outer_iter() {
                n_used.add_assign(&g.slice(s![neigh[0], neigh[1], ..]));
            }
            frep.slice_mut(s![r, c, ..CHANNELS]).assign(&n_used);
        }
    }
    frep
}

/// Given a grid, its feature representation frep,
/// and a set of actions specified by cell, event type and a list of channels,
/// derive feature representations for the afterstates of grid
pub fn incremental_freps(
    grid: &mut Grid,
    frep: &Frep,
    cell: &Cell,
    ce_type: &CEType,
    chs: &[usize],
) -> Freps {
    let (r1, c1) = (cell.row, cell.col);
    let neighs4 = neighbors(4, r1, c1, false);
    let neighs2 = neighbors(2, r1, c1, true);
    let mut freps = Array::zeros((chs.len(), ROWS, COLS, CHANNELS + 1));
    freps.assign(&frep);
    let mut n_used_neighs_diff: isize = 1;
    let mut n_elig_self_diff: isize = -1;
    if *ce_type == CEType::END {
        n_used_neighs_diff = -1;
        n_elig_self_diff = 1;
    }
    for (i, ch) in chs.into_iter().enumerate() {
        for neigh in neighs4.outer_iter() {
            freps[[i, neigh[0], neigh[1], *ch]] =
                (freps[[i, neigh[0], neigh[1], *ch]] as isize + n_used_neighs_diff) as usize;
        }
        if *ce_type == CEType::END {
            grid[[r1, c1, *ch]] = false;
        }
        for neigh_a in neighs2.outer_iter() {
            let (r2, c2) = (neigh_a[0], neigh_a[1]);
            let neighs = neighbors(2, r2, c2, false);
            let mut not_eligible = grid[[r2, c2, *ch]];
            for neigh_b in neighs.outer_iter() {
                not_eligible |= grid[[neigh_b[0], neigh_b[1], *ch]];
            }
            if !not_eligible {
                freps[[i, neigh_a[0], neigh_a[1], CHANNELS]] =
                    (freps[[i, neigh_a[0], neigh_a[1], CHANNELS]] as isize + n_elig_self_diff)
                        as usize;
            }
        }
        if *ce_type == CEType::END {
            grid[[r1, c1, *ch]] = true;
        }
    }
    freps
}

fn main() {}
