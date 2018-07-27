use std::default::Default;
use rulinalg::matrix::{Matrix, BaseMatrix, BaseMatrixMut};
use rand;
use common::matrix_utils::PartialMatrix;
use common::utils;
use ch03::activation;
use ch04::loss;
use ch06::optimizer::Optimizer;


pub trait Layer {
    fn forward(&mut self, x: &Matrix<f32>) -> Matrix<f32>;
    fn backward(&mut self, dout: &Matrix<f32>) -> Matrix<f32>;
}


pub struct Relu {
    mask: PartialMatrix,
}

impl Relu {
    pub fn new() -> Self {
        Relu { mask: Default::default(), }
    }
}

impl Layer for Relu {
    fn forward(&mut self, x: &Matrix<f32>) -> Matrix<f32> {
        self.mask = PartialMatrix::le(x, 0.0);
        let mut out = utils::copy_matrix(x);
        self.mask.set(&mut out, 0.0);
        
        out
    }
    
    fn backward(&mut self, dout: &Matrix<f32>) -> Matrix<f32> {
        let mut dx = utils::copy_matrix(dout);
        self.mask.set(&mut dx, 0.0);
        
        dx
    }
}


pub struct Sigmoid {
    out: Matrix<f32>,
}

impl Sigmoid {
    pub fn new() -> Self {
        Sigmoid { out: matrix![0.0], }
    }
}

impl Layer for Sigmoid {
    fn forward(&mut self, x: &Matrix<f32>) -> Matrix<f32> {
        self.out = activation::sigmoid(utils::copy_matrix(x));
        
        utils::copy_matrix(&self.out)
    }
    
    fn backward(&mut self, dout: &Matrix<f32>) -> Matrix<f32> {
        let dx = dout.elemul(&(-&self.out + 1.0)).elemul(&self.out);
        
        utils::copy_matrix(&dx)
    }
}


pub struct Affine {
    w: Matrix<f32>,
    b: Matrix<f32>,
    x: Matrix<f32>,
    dw: Matrix<f32>,
    db: Matrix<f32>,
}

impl Affine {
    pub fn new(input_size: usize, output_size: usize, w_init_std: f32) -> Self {
        Affine {
            w: Matrix::from_fn(input_size, output_size, |_, _| rand::random::<f32>() * w_init_std),
            b: Matrix::zeros(1, output_size),
            x: matrix![0.0],
            dw: matrix![0.0],
            db: matrix![0.0],
        }
    }
    
    pub fn learn(&mut self, optimizer: &mut Optimizer) {
        optimizer.update(&mut self.w, &self.dw);
        optimizer.update(&mut self.b, &self.db);
    }
}

impl Layer for Affine {
    fn forward(&mut self, x: &Matrix<f32>) -> Matrix<f32> {
        self.x = utils::copy_matrix(x);
        
        let mut out = x * &self.w;
        for mut row in out.row_iter_mut() {
            *row += &self.b;
        }
        
        out
    }
    
    fn backward(&mut self, dout: &Matrix<f32>) -> Matrix<f32> {
        let dx = dout * self.w.transpose();
        self.dw = self.x.transpose() * dout;
        self.db = Matrix::new(1, dout.cols(), dout.sum_rows().into_iter().collect::<Vec<_>>());
        
        dx
    }
}


pub struct SoftmaxWithLoss {
    y: Matrix<f32>,
    t: Matrix<f32>,
}

impl SoftmaxWithLoss {
    pub fn new() -> Self {
        SoftmaxWithLoss { y: matrix![0.0], t: matrix![0.0], }
    }
    
    pub fn set_label(&mut self, t: &Matrix<f32>) {
        self.t = utils::copy_matrix(t);
    }
}

impl Layer for SoftmaxWithLoss {
    fn forward(&mut self, x: &Matrix<f32>) -> Matrix<f32> {
        self.y = activation::softmax(utils::copy_matrix(x));
        let loss_val = loss::cross_entropy_error(&self.y, &self.t);
        
        matrix![loss_val]
    }
    
    fn backward(&mut self, _: &Matrix<f32>) -> Matrix<f32> {
        (&self.y - &self.t) / (self.t.rows() as f32)
    }
}

