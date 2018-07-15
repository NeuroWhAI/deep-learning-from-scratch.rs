use std::f32;
use rulinalg::matrix::{Matrix, BaseMatrix, BaseMatrixMut};
use rand;
use ch03::activation;
use common::utils;
use super::{gradient, loss as loss_function};

pub struct TwoLayerNet {
    pub w1: Matrix<f32>,
    pub b1: Matrix<f32>,
    pub w2: Matrix<f32>,
    pub b2: Matrix<f32>,
}

impl TwoLayerNet {
    pub fn new(input_size: usize, hidden_size: usize, output_size: usize,
        w_init_std: f32) -> Self {
        
        TwoLayerNet {
            w1: Matrix::from_fn(input_size, hidden_size, |_, _| rand::random::<f32>() * w_init_std),
            b1: Matrix::zeros(1, hidden_size),
            w2: Matrix::from_fn(hidden_size, output_size, |_, _| rand::random::<f32>() * w_init_std),
            b2: Matrix::zeros(1, output_size),
        }
    }
    
    pub fn predict(&self, x: &Matrix<f32>) -> Matrix<f32> {
        let mut a1 = x * &self.w1;
        for mut row in a1.row_iter_mut() {
            *row += &self.b1;
        }
        
        let z1 = activation::sigmoid(a1);
        
        let mut a2 = &z1 * &self.w2;
        for mut row in a2.row_iter_mut() {
            *row += &self.b2;
        }
        
        let y = activation::softmax(a2);
        
        y
    }
    
    pub fn loss(&self, x: &Matrix<f32>, t: &Matrix<f32>) -> f32 {
        let y = self.predict(x);
        loss_function::cross_entropy_error(&y, t)
    }
    
    pub fn accuracy(&self, x: &Matrix<f32>, t: &Matrix<f32>) -> f32 {
        let y = self.predict(x);
        let y = utils::argmax(&y);
        
        let t = utils::argmax(t);
        
        let mut correct = 0;
        
        for (v1, v2) in y.iter().zip(t.iter()) {
            if (v1 - v2).abs() < f32::EPSILON {
                correct += 1;
            }
        }
        
        correct as f32 / t.rows() as f32
    }
    
    pub fn numerical_gradient(&mut self, x: &Matrix<f32>, t: &Matrix<f32>)
        -> (Matrix<f32>, Matrix<f32>, Matrix<f32>, Matrix<f32>) {
        
        let p_net = self as *mut TwoLayerNet;
        unsafe {
            (gradient::numerical_gradient(|_| (*p_net).loss(x, t), &mut (*p_net).w1),
            gradient::numerical_gradient(|_| (*p_net).loss(x, t), &mut (*p_net).b1),
            gradient::numerical_gradient(|_| (*p_net).loss(x, t), &mut (*p_net).w2),
            gradient::numerical_gradient(|_| (*p_net).loss(x, t), &mut (*p_net).b2))
        }
    }
    
    pub fn gradient(&mut self, x: &Matrix<f32>, t: &Matrix<f32>)
        -> (Matrix<f32>, Matrix<f32>, Matrix<f32>, Matrix<f32>) {
        
        let f_batch_size = t.rows() as f32;
        
        
        let mut a1 = x * &self.w1;
        for mut row in a1.row_iter_mut() {
            *row += &self.b1;
        }
        
        let z1 = activation::sigmoid(utils::copy_matrix(&a1));
        
        let mut a2 = &z1 * &self.w2;
        for mut row in a2.row_iter_mut() {
            *row += &self.b2;
        }
        
        let y = activation::softmax(a2);
        
        
        let dy = (y - t) / f_batch_size;
        let grad_w2 = &z1.transpose() * &dy;
        let grad_b2 = Matrix::new(1, self.b2.cols(), dy.sum_rows().into_iter().collect::<Vec<_>>());
        
        let da1 = &dy * &self.w2.transpose();
        let dz1 = (-activation::sigmoid(utils::copy_matrix(&a1)) + 1.0) * activation::sigmoid(a1);
        let dz1 = dz1.elemul(&da1);
        let grad_w1 = &x.transpose() * &dz1;
        let grad_b1 = Matrix::new(1, self.b1.cols(), dz1.sum_rows().into_iter().collect::<Vec<_>>());
        
        (grad_w1, grad_b1, grad_w2, grad_b2)
    }
}
