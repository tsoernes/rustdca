mod eventgen;

mod gridfuncs;

#[macro_use]
extern crate ndarray;
#[macro_use]
extern crate lazy_static;
extern crate ordered_float;
extern crate revord;

fn main() {
    let neighs = gridfuncs::neighbors(2, 3, 4, false);
    println!("{}", neighs);
    println!("{}", gridfuncs::CEType::NEW);
    println!("{}", gridfuncs::CEType::NEW == gridfuncs::CEType::NEW);
    println!("{}", gridfuncs::CEType::NEW == gridfuncs::CEType::END);
    println!("{}", gridfuncs::CEType::NEW <= gridfuncs::CEType::END);
}
