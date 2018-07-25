mod gradient;
pub mod loss;
mod two_layer_net;

use rulinalg::matrix::{Matrix, BaseMatrix};
use common::{plot, mnist::Mnist};
use self::two_layer_net::TwoLayerNet;

fn function_2(x: &Matrix<f32>) -> f32 {
    let x1 = x[[0, 0]];
    let x2 = x[[0, 1]];
    return x1 * x1 + x2 * x2;
}

fn test_gradient() {
    let init_x = matrix![-3.0, 4.0f32];
    
    println!("x = {}", init_x);
    println!("f(x) = {}", function_2(&init_x));
    
    let lr = 0.1;
    let step_num = 20;
    
    let (x, history) = gradient::gradient_descent(function_2, &init_x, lr, step_num);
    
    plot::print_graph(&history[..], 50, 20);
    
    println!("x = {}", x);
    println!("f(x) = {}", function_2(&x));
}

fn test_loss() {
    let y = matrix![0.2, 0.1, 0.7;
        1.0, 0.0, 0.0;
        0.4, 0.5, 0.1f32];
    let t = matrix![0.0, 0.0, 1.0;
        1.0, 0.0, 0.0;
        1.0, 0.0, 0.0];
        
    println!("MSE: {}", loss::mean_squared_error(&y, &t));
    println!("CEE: {}", loss::cross_entropy_error(&y, &t));
}

fn test_net() {
    let mnist = Mnist::new();
    let mut net = TwoLayerNet::new(784, 100, 10, 0.01);
    
    let iters_num = 100;
    let train_size = mnist.train_x.rows() / 100;
    let batch_size = 100;
    let learning_rate = 0.1;
    
    for _ in 0..iters_num {
        let mut step = 0.0;
        let mut loss = 0.0;
        let mut acc = 0.0;
    
        let mut batch_offset = 0;
        
        while batch_offset < train_size {
            let batch_range = (batch_offset..(batch_offset + batch_size).min(train_size))
                .collect::<Vec<_>>();
            let batch_x = mnist.train_x.select_rows(&batch_range[..]);
            let batch_y = mnist.train_y.select_rows(&batch_range[..]);
                
            step += 1.0;
            loss += net.loss(&batch_x, &batch_y);
            acc += net.accuracy(&batch_x, &batch_y);
            
            //let (w1, b1, w2, b2) = net.numerical_gradient(&batch_x, &batch_y);
            let (w1, b1, w2, b2) = net.gradient(&batch_x, &batch_y);
            
            net.w1 -= w1 * learning_rate;
            net.b1 -= b1 * learning_rate;
            net.w2 -= w2 * learning_rate;
            net.b2 -= b2 * learning_rate;
            
            batch_offset += batch_size;
        }
        
        println!("Loss: {}, Acc: {}", loss / step, acc / step);
    }
}

pub fn tests() {
    println!("[Gradient]");
    test_gradient();
    
    println!("[Loss]");
    test_loss();
    
    println!("[Net] (May take long time)");
    test_net();
}
