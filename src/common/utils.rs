use rulinalg::matrix::{Matrix, BaseMatrix};

pub fn copy_matrix(m: &Matrix<f32>) -> Matrix<f32> {
    Matrix::new(m.rows(), m.cols(), m.iter().map(|v| *v).collect::<Vec<_>>())
}

pub fn argmax(x: &Matrix<f32>) -> Matrix<f32> {
    Matrix::from_fn(x.rows(), 1, |_, row| {
        let mut max_val = 0.0;
        let mut max_index = -1i32;
    
        for (i, val) in x.row(row).iter().enumerate() {
            if max_index < 0 || *val > max_val {
                max_val = *val;
                max_index = i as i32;
            }
        }
        
        max_index as f32
    })
}
