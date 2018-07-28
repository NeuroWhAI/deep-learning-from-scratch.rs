use rulinalg::matrix::{Matrix, BaseMatrix};
use common::utils;
use ch06::optimizer::Optimizer;
use super::layers::{self, Layer};


pub struct MultiLayerNet {
    affine1: layers::Affine,
    relu1: layers::Relu,
    affine2: layers::Affine,
    
    last: layers::SoftmaxWithLoss,
}

fn get_w_std(w_initializer: &str, input_size: usize) -> f32 {
    match w_initializer {
        "rand" | "random" => 0.01,
        "xavier" => (1.0 / input_size as f32).sqrt(),
        "he" => (2.0 / input_size as f32).sqrt(),
        _ => 0.01
    }
}

impl MultiLayerNet {
    pub fn new(input_size: usize, hidden_size: usize, output_size: usize, w_initializer: &str) -> Self {
        MultiLayerNet {
            affine1: layers::Affine::new(input_size, hidden_size, get_w_std(w_initializer, input_size)),
            relu1: layers::Relu::new(),
            affine2: layers::Affine::new(hidden_size, output_size, get_w_std(w_initializer, hidden_size)),
            
            last: layers::SoftmaxWithLoss::new(),
        }
    }
    
    pub fn predict(&mut self, x: &Matrix<f32>) -> Matrix<f32> {
        let layers: Vec<&mut Layer> = vec![&mut self.affine1, &mut self.relu1, &mut self.affine2];
    
        let mut prev_out = utils::copy_matrix(x);
        
        for layer in layers {
            prev_out = layer.forward(&prev_out);
        }
        
        prev_out
    }
    
    pub fn loss(&mut self, x: &Matrix<f32>, t: &Matrix<f32>) -> f32 {
        let y = self.predict(x);
        
        self.last.set_label(t);
        
        for v in self.last.forward(&y).iter() {
            return *v;
        }
        
        0.0
    }
    
    pub fn accuracy(&mut self, x: &Matrix<f32>, t: &Matrix<f32>) -> f32 {
        let y = self.predict(x);
        let y = utils::argmax(&y);
        let t = utils::argmax(t);
        
        let mut cnt = 0;
        
        for (a, b) in y.iter().zip(t.iter()) {
            if (a - b).abs() < 0.000001 {
                cnt += 1;
            }
        }
        
        cnt as f32 / t.rows() as f32
    }
    
    pub fn learn(&mut self, x: &Matrix<f32>, t: &Matrix<f32>, optimizer: &mut Optimizer) -> f32 {
        // Forward
        let loss_val = self.loss(x, t);
    
    
        // Backward
        {
            let layers: Vec<&mut Layer> = vec![&mut self.affine1, &mut self.relu1, &mut self.affine2];

            let mut dout = matrix![1.0];
            dout = self.last.backward(&dout);

            for layer in layers.into_iter().rev() {
                dout = layer.backward(&dout);
            }
        }
        
        
        // Learn
        self.affine1.learn(optimizer);
        self.affine2.learn(optimizer);
        
        
        loss_val
    }
}

