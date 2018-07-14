use rulinalg::matrix::BaseMatrix;

pub fn and_gate(x1: f32, x2: f32) -> f32 {
	let x = matrix![x1, x2];
	let w = matrix![0.5, 0.5];
	let b = -0.7;
	
	let out = x.elemul(&w).sum() + b;
	
	if out > 0.0 {
		1.0
	}
	else {
		0.0
	}
}

pub fn or_gate(x1: f32, x2: f32) -> f32 {
	let x = matrix![x1, x2];
	let w = matrix![0.5, 0.5];
	let b = -0.2;
	
	let out = x.elemul(&w).sum() + b;
	
	if out > 0.0 {
		1.0
	}
	else {
		0.0
	}
}

pub fn nand_gate(x1: f32, x2: f32) -> f32 {
	let x = matrix![x1, x2];
	let w = matrix![-0.5, -0.5];
	let b = 0.7;
	
	let out = x.elemul(&w).sum() + b;
	
	if out > 0.0 {
		1.0
	}
	else {
		0.0
	}
}

pub fn xor_gate(x1: f32, x2: f32) -> f32 {
	let s1 = nand_gate(x1, x2);
	let s2 = or_gate(x1, x2);
	and_gate(s1, s2)
}
