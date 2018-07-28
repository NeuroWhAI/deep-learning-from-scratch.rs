pub mod optimizer;

use ch05;

pub fn tests() {
    println!("[Optimizer - SGD]");
    ch05::test_layered_net(&mut optimizer::SGD::new(0.01), "random");
    
    println!("[Optimizer - Momentum]");
    ch05::test_layered_net(&mut optimizer::Momentum::new(0.01, 0.9), "random");
    
    println!("[Optimizer - AdaGrad]");
    ch05::test_layered_net(&mut optimizer::AdaGrad::new(0.01), "random");
    
    println!("[W Std : 0.01]");
    ch05::test_layered_net(&mut optimizer::SGD::new(0.01), "random");
    
    println!("[W Std : Xavier]");
    ch05::test_layered_net(&mut optimizer::SGD::new(0.01), "xavier");
    
    println!("[W Std : He]");
    ch05::test_layered_net(&mut optimizer::SGD::new(0.01), "he");
}

