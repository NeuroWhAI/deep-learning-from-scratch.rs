mod layer_naive;

use rulinalg::matrix::{Matrix, BaseMatrix};

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

pub fn tests() {
    println!("[Buy apple and orange]");
    test_layer_naive();
}
