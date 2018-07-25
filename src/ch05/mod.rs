mod layer_naive;
mod layers;
mod multi_layer_net;

use rulinalg::matrix::{Matrix, BaseMatrix};
use common::mnist::Mnist;
use self::multi_layer_net::MultiLayerNet;

fn test_layer_naive() {
    use self::layer_naive::{MulLayer, AddLayer};

    let apple = 100.0;
    let apple_num = 2.0;
    let orange = 150.0;
    let orange_num = 3.0;
    let tax = 1.1;
    
    // Layer
    let mut mul_apple_layer = MulLayer::new();
    let mut mul_orange_layer = MulLayer::new();
    let mut add_all_layer = AddLayer::new();
    let mut mul_tax_layer = MulLayer::new();
    
    // Forward
    let apple_price = mul_apple_layer.forward(apple, apple_num);
    let orange_price = mul_orange_layer.forward(orange, orange_num);
    let all_price = add_all_layer.forward(apple_price, orange_price);
    let price = mul_tax_layer.forward(all_price, tax);
    
    println!("Total apple price: {}", apple_price);
    println!("Total orange price: {}", orange_price);
    println!("Total price without tax: {}", all_price);
    println!("Total price with tax: {}", price);
    
    // Backward
    let d_price = 1.0;
    let (d_all_price, d_tax) = mul_tax_layer.backward(d_price);
    let (d_apple_price, d_orange_price) = add_all_layer.backward(d_all_price);
    let (d_orange, d_orange_num) = mul_orange_layer.backward(d_orange_price);
    let (d_apple, d_apple_num) = mul_apple_layer.backward(d_apple_price);
    
    println!("dApple: {}", d_apple);
    println!("dApple_num: {}", d_apple_num);
    println!("dOrange: {}", d_orange);
    println!("dOrange_num: {}", d_orange_num);
    println!("dTax: {}", d_tax);
}

pub fn test_layered_net() {
    let mnist = Mnist::new();
    let mut net = MultiLayerNet::new(784, 100, 10);
    
    let iters_num = 20;
    let train_size = mnist.train_x.rows();
    let batch_size = 100;
    let learning_rate = 0.1;
    
    for _ in 0..iters_num {
        let mut batch_offset = 0;
        
        let mut loss = 0.0;
        
        while batch_offset < train_size {
            let (batch_x, batch_y) = mnist.get_train_batch(batch_offset, batch_size);
            
            loss += net.learn(&batch_x, &batch_y, learning_rate);
            
            batch_offset += batch_size;
        }
        
        let (train_x, train_y) = mnist.get_train_batch(0, 1000);
        let (val_x, val_y) = mnist.get_validation_batch(0, 1000);
        
        let acc_train = net.accuracy(&train_x, &train_y);
        let acc_val = net.accuracy(&val_x, &val_y);
        
        println!("Loss: {}, Acc: {}, Test Acc: {}", loss, acc_train, acc_val);
    }
    
    let (test_x, test_y) = mnist.get_test_batch(0, 1000);
    let acc_test = net.accuracy(&test_x, &test_y);
        
    println!("Final test acc: {}", acc_test);
}

pub fn tests() {
    println!("[Buy apple and orange]");
    test_layer_naive();
    
    println!("[Layered Network]");
    test_layered_net();
}
