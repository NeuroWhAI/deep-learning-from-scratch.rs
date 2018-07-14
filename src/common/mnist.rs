use mnist::MnistBuilder;
use rulinalg::matrix::Matrix;

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
}
