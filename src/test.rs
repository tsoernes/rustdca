fn foo(input: Option<i32>) -> Option<i32> {
    if input.is_none() {
        return None;
    }

    let input = input.unwrap();
    if input < 0 {
        return None;
    }
    Some(input)
}

fn foo2(input: Option<i32>) -> Option<i32> {
    match input {
        Some(i) if i >= 0 => Some(i),
        _ => None
    }
}

fn main() {
    let xs = [None, Some(-1), Some(0), Some(1), Some(5)];
    for x in &xs {
        print!("Foo1: ");
        let fx1 = foo(*x);
        match fx1 {
            Some(fx1) => println!("Some {}", fx1),
            None => println!("None")
        }
        let fx2 = foo2(*x);
        println!("Eq: {}", fx1 == fx2);
        assert_eq!(fx1, fx2);

        let bx1 = bar(*x);
        let bx2 = bar2(*x);
        assert_eq!(bx1, bx2);
    }
}


#[derive(Eq, PartialEq, Debug)]
enum ErrNegative {
    V1,
    V2
}

fn bar(input: Option<i32>) -> Result<i32, ErrNegative> {
    match foo(input) {
        Some(n) => Ok(n),
        None => Err(ErrNegative::V1)
    }
}

fn bar2(input: Option<i32>) -> Result<i32, ErrNegative> {
    foo(input).ok_or(ErrNegative::V1)
}
