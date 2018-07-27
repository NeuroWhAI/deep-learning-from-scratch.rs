use std::collections::HashMap;
use rulinalg::matrix::{Matrix, BaseMatrix, BaseMatrixMut};
use common::utils;


pub trait Optimizer {
    fn update(&mut self, param: &mut Matrix<f32>, grad: &Matrix<f32>);
}


pub struct SGD {
    lr: f32,
}

impl SGD {
    pub fn new(lr: f32) -> Self {
        SGD { lr: lr }
    }
}

impl Optimizer for SGD {
    fn update(&mut self, param: &mut Matrix<f32>, grad: &Matrix<f32>) {
        *param -= grad * self.lr;
    }
}


pub struct Momentum {
    lr: f32,
    momentum: f32,
    v: HashMap<*const Matrix<f32>, Matrix<f32>>,
}

impl Momentum {
    pub fn new(lr: f32, momentum: f32) -> Self {
        Momentum {
            lr: lr,
            momentum: momentum,
            v: HashMap::new(),
        }
    }
}

impl Optimizer for Momentum {
    fn update(&mut self, param: &mut Matrix<f32>, grad: &Matrix<f32>) {
        let key = param as *const Matrix<f32>;
    
        if !self.v.contains_key(&key) {
            self.v.insert(key, Matrix::zeros(param.rows(), param.cols()));
        }
        
        if let Some(velocity) = self.v.get_mut(&key) {
            // NOTE: &*velocity : &mut T -> &T
            *velocity = &*velocity * self.momentum - grad * self.lr;
            *param += &*velocity;
        }
    }
}


pub struct AdaGrad {
    lr: f32,
    h: HashMap<*const Matrix<f32>, Matrix<f32>>,
}

impl AdaGrad {
    pub fn new(lr: f32) -> Self {
        AdaGrad {
            lr: lr,
            h: HashMap::new(),
        }
    }
}

impl Optimizer for AdaGrad {
    fn update(&mut self, param: &mut Matrix<f32>, grad: &Matrix<f32>) {
        let key = param as *const Matrix<f32>;
    
        if !self.h.contains_key(&key) {
            self.h.insert(key, Matrix::zeros(param.rows(), param.cols()));
        }
        
        if let Some(h) = self.h.get_mut(&key) {
            *h += grad.elemul(grad);
            
            // NOTE: &*h : &mut T -> &T
            let sqrt_h = utils::copy_matrix(&*h)
                .apply(&|value| value.sqrt() + 1e-7);
            *param -= (grad * self.lr).elediv(&sqrt_h);
        }
    }
}

