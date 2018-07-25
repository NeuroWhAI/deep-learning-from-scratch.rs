use mnist::MnistBuilder;
use rulinalg::matrix::{Matrix, BaseMatrix};

pub struct Mnist {
    pub train_x: Matrix<f32>,
    pub train_y: Matrix<f32>,
    pub validation_x: Matrix<f32>,
    pub validation_y: Matrix<f32>,
    pub test_x: Matrix<f32>,
    pub test_y: Matrix<f32>,
}

impl Mnist {
    pub fn new() -> Mnist {
        let train_size = 50_000;
        let val_size = 10_000;
        let test_size = 10_000;
    
        // Deconstruct the returned Mnist struct.
        let mnist = MnistBuilder::new()
            .base_path("mnist")
            .label_format_one_hot()
            .training_set_length(train_size)
            .validation_set_length(val_size)
            .test_set_length(test_size)
            .finalize();
    
        fn convert(data: &Vec<u8>, width: usize, height: usize) -> Matrix<f32> {
            Matrix::new(width, height,
                data.iter().map(|v| *v as f32).collect::<Vec<_>>())
        }
    
        Mnist {
            train_x: convert(&mnist.trn_img, train_size as usize, 784) / 255.0,
            train_y: convert(&mnist.trn_lbl, train_size as usize, 10),
            validation_x: convert(&mnist.val_img, val_size as usize, 784) / 255.0,
            validation_y: convert(&mnist.val_lbl, val_size as usize, 10),
            test_x: convert(&mnist.tst_img, test_size as usize, 784) / 255.0,
            test_y: convert(&mnist.tst_lbl, test_size as usize, 10),
        }
    }
    
    fn get_batch(x: &Matrix<f32>, y: &Matrix<f32>, offset: usize, batch_size: usize)
        -> (Matrix<f32>, Matrix<f32>) {
        
        let batch_range = (offset..(offset + batch_size).min(y.rows()))
                .collect::<Vec<_>>();
        let batch_x = x.select_rows(&batch_range[..]);
        let batch_y = y.select_rows(&batch_range[..]);
        
        (batch_x, batch_y)
    }
    
    pub fn get_train_batch(&self, offset: usize, batch_size: usize) -> (Matrix<f32>, Matrix<f32>) {
        Mnist::get_batch(&self.train_x, &self.train_y, offset, batch_size)
    }
    
    pub fn get_validation_batch(&self, offset: usize, batch_size: usize) -> (Matrix<f32>, Matrix<f32>) {
        Mnist::get_batch(&self.validation_x, &self.validation_y, offset, batch_size)
    }
    
    pub fn get_test_batch(&self, offset: usize, batch_size: usize) -> (Matrix<f32>, Matrix<f32>) {
        Mnist::get_batch(&self.test_x, &self.test_y, offset, batch_size)
    }
}
