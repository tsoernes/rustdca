use eventgen::EType;
use ndarray::prelude::*;
use ndarray::{Data, DataMut};
use std::ops::AddAssign;
use std::ops::BitAnd;
use std::ops::BitOr;
use std::ops::BitOrAssign;
use std::ops::Not;

pub const ROWS: usize = 7;
pub const COLS: usize = 7;
pub const CHANNELS: usize = 70;

// TODO NOTE
// Consider doing all ops on freps as usize and casting to f32 at end only

lazy_static! {
    // (NEIGHS1, NEIGHS2, NEIGHS4, N_NEIGHS):
    static ref NEIGHS: (Array<usize, Ix4>, Array<usize, Ix4>,
                        Array<usize, Ix4>, Array<usize, Ix3>) = generate_neighs();
}

#[derive(Eq, PartialEq, Hash, Clone, Debug)]
pub struct Cell {
    pub row: usize,
    pub col: usize,
}

pub type Grid<S: Data<Elem = bool>> = ArrayBase<S, Ix3>;
pub type GridO = Array<bool, Ix3>;
pub type GridsO = Array<bool, Ix4>;
pub type Frep<S: Data<Elem = f32>> = ArrayBase<S, Ix3>;
pub type FrepO = Array<f32, Ix3>;
pub type FrepsO = Array<f32, Ix4>;

/// Distance from cell (r1, c1) to cell (r2, c2) in a hexagonal grid
fn hex_distance(r1: i8, c1: i8, r2: i8, c2: i8) -> i8 {
    ((r1 - r2).abs() + (r1 + c1 - r2 - c2).abs() + (c1 - c2).abs()) / 2
}

/// Return 3 arrays with indecies of neighbors (including self) within distance of 1, 2, and 4.
/// For a given distance, cells have a varying number of neighbors across the grid since neighbors
/// outside of grid boundaries do not count.
/// Therefore, a (3 x Rows x Cols) array is also returned, which contains,
/// for each of the distances above, for each cell, the number of neighbors with the given
/// distance or less.
fn generate_neighs() -> (
    Array<usize, Ix4>,
    Array<usize, Ix4>,
    Array<usize, Ix4>,
    Array<usize, Ix3>,
) {
    // Indecies of neighbors with distance of 1 or less
    let mut neighs1 = Array::zeros((ROWS, COLS, 7, 2));
    // Indecies of neighbors with distance of 2 or less
    let mut neighs2 = Array::zeros((ROWS, COLS, 19, 2));
    // Indecies of neighbors with distance of 4 or less
    let mut neighs4 = Array::zeros((ROWS, COLS, 43, 2));
    // For each distance above, for each row and col:
    // the number of neighbors with the given distance or less.
    let mut n_neighs = Array::zeros((3, ROWS, COLS));
    for r1 in 0..ROWS {
        for c1 in 0..COLS {
            // Store index of focal cell first, so that it can be easily excluded
            neighs1.slice_mut(s![r1, c1, 0, ..]).assign(&array![r1, c1]);
            neighs2.slice_mut(s![r1, c1, 0, ..]).assign(&array![r1, c1]);
            neighs4.slice_mut(s![r1, c1, 0, ..]).assign(&array![r1, c1]);
            // Because 'n_neighs' defines
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

/// Indecies of neighbors with distance 'dist' or less from the cell
/// specified by (row, col).
/// The indecies are of the format:
/// [[row1, row2, .., rowN], [col1, col2, .., colN]] for N neighbors
/// where (row1, col1) equals (row, col) if 'include_self' is True,
/// else it is an arbitrary neighbor.
pub fn neighbors(
    dist: usize,
    row: usize,
    col: usize,
    include_self: bool,
) -> ArrayView<'static, usize, Ix2> {
    let (allneighs, d) = match dist {
        1 => (&NEIGHS.0, 0),
        2 => (&NEIGHS.1, 1),
        4 => (&NEIGHS.2, 2),
        _ => panic!("Neighbors for distances other than 1, 2 or 4 should never be needed"),
    };
    let start = if include_self { 0 } else { 1 };
    let end = NEIGHS.3[[d, row, col]];
    debug_assert_ne!(
        allneighs.slice(s![row, col, end, ..]),
        Array::from_vec(vec![0 as usize, 0]),
        "Neighs: {:?}\nEnd: {}",
        allneighs.slice(s![row, col, ..end, ..]),
        end
    );
    allneighs.slice(s![row, col, start..end, ..])
}

pub fn cell_of(neighs: ArrayView<usize, Ix2>, i: usize) -> Cell {
    let neigh = neighs.slice(s![i, ..]);
    Cell {
        row: neigh[[0]],
        col: neigh[[1]],
    }
}

/// Alloc. map of channels in use at cell neighbors with distance of 2 or less
fn inuse_neighs<S: Data<Elem = bool>>(grid: &Grid<S>, cell: &Cell) -> Array<bool, Ix1> {
    let neighs = neighbors(2, cell.row, cell.col, false);
    let mut alloc_map: Array1<bool> = Array::default(CHANNELS);
    for neigh in neighs.outer_iter() {
        alloc_map.bitor_assign(&grid.slice(s![neigh[0], neigh[1], ..]));
    }
    alloc_map
}

/// One-hot array of eligible channels
fn eligible_map<S: Data<Elem = bool>>(grid: &Grid<S>, cell: &Cell) -> Array<bool, Ix1> {
    inuse_neighs(grid, cell)
        .bitor(&grid.slice(s![cell.row, cell.col, ..]))
        .not()
}

pub fn get_inuse_chs<S: Data<Elem = bool>>(grid: &Grid<S>, cell: &Cell) -> Vec<usize> {
    grid.slice(s![cell.row, cell.col, ..])
        .indexed_iter()
        .filter_map(|(index, &item)| if item { Some(index) } else { None })
        .collect()
}

/// Return the eligible channels for the given cell. A channel is eligible if it is free at
/// the cell and all of its neighbors with distance of 2 or less.
pub fn get_eligible_chs<S: Data<Elem = bool>>(grid: &Grid<S>, cell: &Cell) -> Vec<usize> {
    eligible_map(grid, cell)
        .indexed_iter()
        .filter_map(|(index, &item)| if item { Some(index) } else { None })
        .collect()
}

/// Return Some(argmax, max) of a 1D array; None if its empty
pub fn argpmax1<N: PartialOrd + Copy>(arr: &Array1<N>) -> Option<(usize, N)> {
    arr.indexed_iter()
        .fold(None, |acc, (idx, &elem)| match acc {
            Some((acc_idx, acc_elem)) if acc_elem > elem => Some((acc_idx, acc_elem)),
            _ => Some((idx, elem)),
        })
}

/// Given a grid 'grid' and a set of actions, the latter specified by a cell, an event type
/// and a list of channels, return the grids ('afterstates') that would result from
/// executing each of the actions on 'grid'
pub fn afterstates<S: Data<Elem = bool>>(
    grid: &Grid<S>,
    cell: &Cell,
    etype: &EType,
    chs: &[usize],
) -> GridsO {
    let targ_val = match etype {
        EType::END => false,
        _ => true,
    };
    let mut grids: GridsO = Array::default((chs.len(), 7, 7, 70));
    for (i, ch) in chs.into_iter().enumerate() {
        grids.slice_mut(s![i, .., .., ..]).assign(&grid);
        grids[[i, cell.row, cell.col, *ch]] = targ_val;
    }
    grids
}

/// Returns an error if the reuse constraint is violated
pub fn validate_reuse_constraint<S: Data<Elem = bool>>(grid: &Grid<S>) -> Result<(), String> {
    for r in 0..ROWS {
        for c in 0..COLS {
            let cell = Cell { row: r, col: c };
            // Channels in use at any neighbor within the reuse distance AND the focal cell 'cell'
            let inuse = inuse_neighs(grid, &cell).bitand(&grid.slice(s![r, c, ..]));
            debug_assert_eq!(inuse.shape(), &[CHANNELS]);
            if inuse.into_iter().any(|&x| x) {
                return Err(format!(
                    "Reuse constraint violated at (r,c): ({}, {})",
                    r, c
                ));
            }
        }
    }
    Ok(())
}

/// A feature representation (frep for short) of the grid. The frep is of the
/// same spatial dimension as the grid. For a given cell, the first 'CHANNELS' features
/// specifies how many times each of the channels is in used within a 4-cell radius,
/// not including the cell itself. An additional feature counts the number of eligible
/// channels in that cell.
pub fn feature_rep<S: Data<Elem = bool>>(grid: &Grid<S>) -> FrepO {
    let mut frep = Array::zeros((ROWS, COLS, CHANNELS + 1));
    for r in 0..ROWS {
        for c in 0..COLS {
            let mut n_used: Array1<u32> = Array::zeros(CHANNELS);
            for neigh in neighbors(4, r, c, false).outer_iter() {
                n_used.add_assign(&grid.slice(s![neigh[0], neigh[1], ..]).mapv(|x| x as u32));
            }
            frep.slice_mut(s![r, c, ..CHANNELS]).assign(&n_used);
            // Find the number of eligible channels for cell (r, c)
            let elig = eligible_map(&grid, &Cell { row: r, col: c });
            frep[[r, c, CHANNELS]] = elig.fold(0, |acc, &x| acc + x as u32);
        }
    }
    frep.mapv(|x: u32| x as f32)
}

/// Given a grid, its feature representation 'frep',
/// and a set of actions specified by cell, event type and a list of channels,
/// derive feature representations for the afterstates of grid.
/// The grid is only temporarily modified.
pub fn incremental_freps<S: Data<Elem = f32>, T: Data<Elem = bool> + DataMut>(
    grid: &mut Grid<T>,
    frep: &Frep<S>,
    cell: &Cell,
    etype: &EType,
    chs: &[usize],
) -> FrepsO {
    // panic!("YIELDS INCORRECT RESULTS");
    let (r1, c1) = (cell.row, cell.col);
    let neighs4 = neighbors(4, r1, c1, false);
    let neighs2 = neighbors(2, r1, c1, true);
    let mut freps = Array::zeros((chs.len(), ROWS, COLS, CHANNELS + 1));
    freps.assign(&frep);
    let mut n_used_neighs_diff: isize = 1;
    let mut n_elig_self_diff: isize = -1;
    if *etype == EType::END {
        n_used_neighs_diff = -1;
        n_elig_self_diff = 1;

        for ch in chs.into_iter() {
            grid[[r1, c1, *ch]] = false;
        }
    }

    for (i, ch) in chs.into_iter().enumerate() {
        for neigh in neighs4.outer_iter() {
            freps[[i, neigh[0], neigh[1], *ch]].add_assign(n_used_neighs_diff as f32);
        }
        for neigh_a in neighs2.outer_iter() {
            let (r2, c2) = (neigh_a[0], neigh_a[1]);
            let neighs = neighbors(2, r2, c2, false);
            // A channel is eligible if it is free in the focal cell ('neigh_a')
            // and every neighbor ('neighs') of the focal cell within the reuse
            // distance (which is 2). Channels in use are 'True' in the grid map matrix
            // so folding the bitwise_or operator over (partial) gridmap of the focal cell
            // and it's co-channel neighbors yields a vector of bits where an index is
            // 'false' if the channel is eligible.
            let mut not_eligible = grid[[r2, c2, *ch]];
            for neigh_b in neighs.outer_iter() {
                not_eligible |= grid[[neigh_b[0], neigh_b[1], *ch]];
            }
            if !not_eligible {
                freps[[i, r2, c2, CHANNELS]].add_assign(n_elig_self_diff as f32);
            }
        }
    }

    if *etype == EType::END {
        for ch in chs.into_iter() {
            grid[[r1, c1, *ch]] = true;
        }
    }
    freps
}

/// Number of channels in use on the whole grid
pub fn n_used<S: Data<Elem = bool>>(grid: &Grid<S>) -> usize {
    grid.fold(0, |n_used, &inuse| n_used + inuse as usize)
}

#[cfg(test)]
mod tests {
    use gridfuncs::*;
    use itertools::free::zip;

    fn eq_frep<S: Data<Elem = f32>, T: Data<Elem = f32>>(frep1: Frep<S>, frep2: Frep<T>) {
        assert_eq!(frep1.shape(), frep2.shape());
        assert_eq!(frep1.shape(), &[ROWS, COLS, CHANNELS + 1]);
        // Check the equality of feature #1 (number of used chs within reuse dist)
        assert_eq!(
            frep1.slice(s![.., .., ..CHANNELS]),
            frep2.slice(s![.., .., ..CHANNELS])
        );
        // Check the equality of feature #2 (number of eligible chs)
        assert_eq!(
            frep1.slice(s![.., .., CHANNELS]),
            frep2.slice(s![.., .., CHANNELS])
        );
    }

    /// Check that deriving feature reps incrementally yields the same result
    /// as doing it from scratch.
    fn incremental_vs_scratch(grid: &mut GridO, cell: &Cell, etype: &EType, chs: &[usize]) {
        let astates = afterstates(&grid, &cell, &etype, &chs);
        let pre_frep = feature_rep(&grid);
        let freps_a = incremental_freps(grid, &pre_frep, &cell, &etype, &chs);
        for (astate, frep_a) in zip(astates.outer_iter(), freps_a.outer_iter()) {
            let frep_b = feature_rep(&astate);
            eq_frep(frep_a, frep_b);
        }
    }

    #[test]
    /// Case: Call arrival on empty grid
    fn test_incremental_1() {
        let mut grid: GridO = Array3::default((ROWS, COLS, CHANNELS));
        let grid_original = grid.clone();
        let cell = Cell { row: 2, col: 3 };
        let etype = EType::NEW;
        let chs = get_eligible_chs(&grid, &cell);
        incremental_vs_scratch(&mut grid, &cell, &etype, &chs);
        // Grid should not have changed
        assert_eq!(grid, grid_original);
    }

    #[test]
    /// Case: Call termination of the only channel in use
    fn test_incremental_2() {
        let mut grid: GridO = Array3::default((ROWS, COLS, CHANNELS));
        grid[[4, 1, 4]] = true;
        let cell = Cell { row: 4, col: 1 };
        let etype = EType::END;
        let chs = get_inuse_chs(&grid, &cell);
        incremental_vs_scratch(&mut grid, &cell, &etype, &chs);
    }

    #[test]
    /// Case: Call arrival when focal cell and neighbor has channels in use
    fn test_incremental_3() {
        let mut grid: GridO = Array3::default((ROWS, COLS, CHANNELS));
        let cell = Cell { row: 0, col: 0 };
        grid[[0, 0, 4]] = true;
        grid[[0, 1, 5]] = true;
        let etype = EType::NEW;
        let chs = get_eligible_chs(&grid, &cell);
        incremental_vs_scratch(&mut grid, &cell, &etype, &chs);
    }

    #[test]
    fn test_feature_rep1() {
        let grid: GridO = Array3::default((ROWS, COLS, CHANNELS));
        let frep = feature_rep(&grid);
        // No cell has a channel in use by any of its neighbors
        let f1_target = Array3::zeros((ROWS, COLS, CHANNELS));
        assert_eq!(frep.slice(s![.., .., ..-1]), f1_target);

        // No cell has any channels in use, i.e. all are free
        let f2_target = Array2::from_elem((ROWS, COLS), CHANNELS as f32);
        assert_eq!(frep.slice(s![.., .., -1]), f2_target);
    }

    #[test]
    fn test_feature_rep2() {
        let mut grid: GridO = Array3::default((ROWS, COLS, CHANNELS));
        grid.slice_mut(s![.., .., 0]).assign(&array!(true));
        let frep = feature_rep(&grid);

        // Every cell has 'n_neighs(cell)' neighbors4 who uses channel 0
        // ('n_neighs(cell)' depends on cell coordinates)
        let mut f1_target = Array3::zeros((ROWS, COLS, CHANNELS));
        for r in 0..ROWS {
            for c in 0..COLS {
                // The number of neighs with dist 4 or less
                let n_neighs = NEIGHS.3[[2, r, c]] - 1;
                f1_target[[r, c, 0]] = n_neighs as f32;
            }
        }
        assert_eq!(frep.slice(s![.., .., ..-1]), f1_target);

        // All cells have the same channel in use, thus all but one of the channels are
        // eligible in all of the cells
        let f2_target = Array2::from_elem((ROWS, COLS), (CHANNELS - 1) as f32);
        assert_eq!(frep.slice(s![.., .., -1]), f2_target);
    }

    #[test]
    fn test_feature_rep3() {
        let mut grid: GridO = Array3::default((ROWS, COLS, CHANNELS));
        let (r, c, ch) = (1, 2, 9);
        grid[[r, c, ch]] = true;
        let frep = feature_rep(&grid);

        // Cell (1, 2) has no neighs that use ch9. The neighs of (1, 2)
        // has 1 neigh that use ch9.
        let mut f1_target = Array3::default((ROWS, COLS, CHANNELS));
        for neigh in neighbors(4, r, c, false).outer_iter() {
            f1_target[[neigh[0], neigh[1], ch]] = 1.0;
        }
        assert_eq!(frep.slice(s![.., .., ..-1]), f1_target);

        // The interfering neighbors of (row, col) = (1, 2) has one less eligible channel
        let mut f2_target = Array2::from_elem((ROWS, COLS), CHANNELS as f32);
        for neigh in neighbors(2, r, c, true).outer_iter() {
            f2_target[[neigh[0], neigh[1]]] -= 1.0;
        }
        assert_eq!(frep.slice(s![.., .., -1]), f2_target);
    }

    #[test]
    fn test_get_eligible_chs() {
        let mut grid: GridO = Array3::from_elem((ROWS, COLS, CHANNELS), true);
        let (r, c) = (3, 4);
        let chs: [usize; 3] = [0, 4, 10];
        for neigh in neighbors(2, r, c, true).outer_iter() {
            for ch in chs.into_iter() {
                grid[[neigh[[0]], neigh[[1]], *ch]] = false
            }
        }
        let elig = get_eligible_chs(&grid, &Cell { row: r, col: c });
        assert_eq!(chs.to_vec(), elig);
    }
}
