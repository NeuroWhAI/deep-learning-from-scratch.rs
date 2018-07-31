pub mod optimizer;

use ch05;

pub fn tests() {
    //println!("[Optimizer - SGD]");
    //ch05::test_layered_net(&mut optimizer::SGD::new(0.01), "random", false);
    
    //println!("[Optimizer - Momentum]");
    //ch05::test_layered_net(&mut optimizer::Momentum::new(0.01, 0.9), "random", false);
    
    println!("[Optimizer - AdaGrad]");
    ch05::test_layered_net(&mut optimizer::AdaGrad::new(0.01), "random", false);
    
    println!("[With dropout]");
    ch05::test_layered_net(&mut optimizer::AdaGrad::new(0.01), "random", true);
    
    //println!("[W Std : 0.01]");
    //ch05::test_layered_net(&mut optimizer::SGD::new(0.01), "random", false);
    
    //println!("[W Std : Xavier]");
    //ch05::test_layered_net(&mut optimizer::SGD::new(0.01), "xavier", false);
    
    //println!("[W Std : He]");
    //ch05::test_layered_net(&mut optimizer::SGD::new(0.01), "he", false);
}

