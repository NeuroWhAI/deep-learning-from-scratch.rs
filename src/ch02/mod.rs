mod perceptron;

fn test_gate(name: &str, gate: fn(f32, f32) -> f32) {
    let input = [
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0],
    ];
    
    println!("{}", name);
    
    for x in input.iter() {
        let x1 = x[0];
        let x2 = x[1];

        println!("({}, {}) -> {}", x1, x2, gate(x1, x2));
    }
}

pub fn tests() {
    test_gate("AND", perceptron::and_gate);
    test_gate("OR", perceptron::or_gate);
    test_gate("NAND", perceptron::nand_gate);
    test_gate("XOR", perceptron::xor_gate);
}
