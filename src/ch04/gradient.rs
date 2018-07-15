use rulinalg::matrix::{Matrix, BaseMatrix, BaseMatrixMut};

pub fn numerical_gradient<F>(f: F, x: &mut Matrix<f32>) -> Matrix<f32>
    where F: Fn(&Matrix<f32>) -> f32 {
    
    let h = 1e-4;
    let mut grad = Matrix::<f32>::zeros(x.rows(), x.cols());
    
    for it in x.iter_mut().zip(grad.iter_mut()) {
        let (v, g) = it;
    
        let bak = *v;
        
        *v = bak + h;
        let fxh1 = f(x);
        
        *v = bak - h;
        let fxh2 = f(x);
        
        *g = (fxh1 - fxh2) / (2.0 * h);
        *v = bak;
    }
    
    grad
}

pub fn gradient_descent(f: fn(&Matrix<f32>) -> f32, init_x: &Matrix<f32>,
    lr: f32, step_num: usize) -> (Matrix<f32>, Vec<f32>) {
    
    let mut x = Matrix::new(init_x.rows(), init_x.cols(),
        init_x.iter().map(|v| *v).collect::<Vec<_>>());
    let mut history = Vec::new();
        
    for _ in 0..step_num {
        let grad = numerical_gradient(f, &mut x);
        x -= grad * lr;
        
        history.push(f(&x));
    }
    
    (x, history)
}
