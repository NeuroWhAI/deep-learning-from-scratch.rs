pub mod optimizer;

use ch05;

pub fn tests() {
    println!("[Optimizer - SGD]");
    ch05::test_layered_net(&mut optimizer::SGD::new(0.01));
    
    println!("[Optimizer - Momentum]");
    ch05::test_layered_net(&mut optimizer::Momentum::new(0.01, 0.9));
    
    println!("[Optimizer - AdaGrad]");
    ch05::test_layered_net(&mut optimizer::AdaGrad::new(0.01));
}

