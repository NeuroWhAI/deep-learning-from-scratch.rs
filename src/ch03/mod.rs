mod activation;

use rulinalg::matrix::{Matrix, BaseMatrix};
use common::plot;

fn test_function(f: fn(Matrix<f32>) -> Matrix<f32>) {
	let data: Vec<_> = (-50..50).map(|n| n as f32 / 10.0).collect();
	let out = f(Matrix::new(1, data.len(), data));
	
	let graph_data: Vec<_> = out.iter().map(|v| *v).collect();
	plot::print_graph(&graph_data[..], 50, 20);
}

pub fn tests() {
	println!("Step function");
	test_function(activation::step_function);
	
	println!("Sigmoid");
	test_function(activation::sigmoid);
	
	println!("ReLU");
	test_function(activation::relu);
}
