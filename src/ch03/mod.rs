mod activation;

pub fn tests() {
	println!("Step function");
	for x in -2..3 {
		println!("{} -> {}", x, activation::step_function(matrix![x as f32]));
	}
	
	println!("Sigmoid");
	for x in -2..3 {
		println!("{} -> {}", x, activation::sigmoid(matrix![x as f32]));
	}
	
	println!("ReLU");
	for x in -2..3 {
		println!("{} -> {}", x, activation::relu(matrix![x as f32]));
	}
}
