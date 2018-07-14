use std::f32;

pub fn print_graph(data: &[f32], width: usize, height: usize) {
	let f_height = height as f32;


	// Allocate buffer.
	let mut buffer = Vec::new();
	buffer.resize(height, Vec::new());
	
	for line in &mut buffer {
		line.resize(width, ' ');
	}
	
	
	// Calculate graph ratio.
	let max_val = data.iter().map(|v| *v).fold(f32::NAN, f32::max);
	let min_val = data.iter().map(|v| *v).fold(f32::NAN, f32::min);
	
	let value_gap = f32::max(max_val - min_val, 1.0);
	
	let x_per_value = width as f32 / data.len() as f32;
	
	
	// Draw graph.
	let mut skip_gage = 0.0;
	let mut x = 0;
	
	for val in data {
		skip_gage += x_per_value;
	
		if skip_gage >= 1.0 {
			skip_gage -= 1.0;
		
			let y = f32::floor(f_height - 1.0 - (val - min_val) / value_gap * (f_height - 1.0)) as usize;
		
			buffer[y][x] = '-';
			
			x += 1;
			if x >= width {
				break;
			}
		}
	}
	
	
	// Print graph.
	for line in &buffer {
		let line_str: String = line.iter().collect();
		println!("{}", line_str);
	}
}
