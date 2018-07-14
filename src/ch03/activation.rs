use rulinalg::matrix::{Matrix, BaseMatrixMut};

pub fn step_function(x: Matrix<f32>) -> Matrix<f32> {
	x.apply(&|value| if value > 0.0 { 1.0 } else { 0.0 })
}

pub fn sigmoid(x: Matrix<f32>) -> Matrix<f32> {
	x.apply(&|value| 1.0 / (1.0 + (-value).exp()))
}

pub fn relu(x: Matrix<f32>) -> Matrix<f32> {
	x.apply(&|value| if value > 0.0 { value } else { 0.0 })
}
