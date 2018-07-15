use rulinalg::matrix::{Matrix, BaseMatrix, BaseMatrixMut, Axes};

pub fn step_function(x: Matrix<f32>) -> Matrix<f32> {
    x.apply(&|value| if value > 0.0 { 1.0 } else { 0.0 })
}

pub fn sigmoid(x: Matrix<f32>) -> Matrix<f32> {
    x.apply(&|value| 1.0 / (1.0 + (-value).exp()))
}

pub fn relu(x: Matrix<f32>) -> Matrix<f32> {
    x.apply(&|value| if value > 0.0 { value } else { 0.0 })
}

pub fn softmax(mut x: Matrix<f32>) -> Matrix<f32> {
    
    for mut col in x.col_iter_mut() {
        let max = col.max(Axes::Col)[0];
        
        let mut sum_exp = 0.0;
        for val in col.iter_mut() {
            *val = (*val - max).exp();
            sum_exp += *val;
        }
        
        for val in col.iter_mut() {
            *val /= sum_exp
        }
    }
    
    x
}
