use rulinalg::matrix::{Matrix, BaseMatrix, BaseMatrixMut};

pub fn mean_squared_error(y: &Matrix<f32>, t: &Matrix<f32>) -> f32 {
    let mut err = y - t;
    err = err.apply(&|v| v * v);
    
    err.sum() * 0.5
}

pub fn cross_entropy_error(y: &Matrix<f32>, t: &Matrix<f32>) -> f32 {
    let temp = y + 1e-7;
    let temp = t.elemul(&temp.apply(&|v| v.log(10.0)));
    
    -temp.sum() / y.rows() as f32
}
