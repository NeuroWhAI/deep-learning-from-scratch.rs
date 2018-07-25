use rulinalg::matrix::{Matrix, BaseMatrix, BaseMatrixMut};
use std::default::Default;


pub struct PartialMatrix {
    indices: Vec<(usize, usize)>,
}

impl PartialMatrix {
    pub fn new() -> Self {
        Default::default()
    }

    pub fn filter<F>(m: &Matrix<f32>, func: F) -> Self
        where F: Fn(&f32) -> bool {
        
        let mut indices: Vec<(usize, usize)> = Vec::new();
        
        for (y, row) in m.row_iter().enumerate() {
            for (x, v) in row.iter().enumerate() {
                if func(v) {
                    indices.push((y, x));
                }
            }
        }
        
        PartialMatrix { indices: indices, }
    }

    pub fn lt(m: &Matrix<f32>, val: f32) -> Self {
        PartialMatrix::filter(m, |&v| v < val)
    }

    pub fn le(m: &Matrix<f32>, val: f32) -> Self {
        PartialMatrix::filter(m, |&v| v <= val)
    }

    pub fn gt(m: &Matrix<f32>, val: f32) -> Self {
        PartialMatrix::filter(m, |&v| v > val)
    }

    pub fn ge(m: &Matrix<f32>, val: f32) -> Self {
        PartialMatrix::filter(m, |&v| v >= val)
    }
}

impl PartialMatrix {
    pub fn apply<F>(&self, m: &mut Matrix<f32>, func: F)
        where F: Fn(&f32) -> f32 {
    
        for index in &self.indices {
            if let Some(target) = m.row_mut(index.0).col_mut(index.1).iter_mut().next() {
                *target = func(target);
            }
        }
    }

    pub fn set(&self, m: &mut Matrix<f32>, val: f32) {
        self.apply(m, |_| val);
    }

    pub fn add(&self, m: &mut Matrix<f32>, val: f32) {
        self.apply(m, |&v| v + val);
    }

    pub fn sub(&self, m: &mut Matrix<f32>, val: f32) {
        self.apply(m, |&v| v - val);
    }
}

impl Default for PartialMatrix {
    fn default() -> Self {
        PartialMatrix { indices: Default::default() }
    }
}

