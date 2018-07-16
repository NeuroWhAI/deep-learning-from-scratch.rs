pub struct MulLayer {
    x: f32,
    y: f32,
}

impl MulLayer {
    pub fn new() -> Self {
        MulLayer {
            x: 0.0,
            y: 0.0
        }
    }
    
    pub fn forward(&mut self, x: f32, y: f32) -> f32 {
        self.x = x;
        self.y = y;
        
        x * y
    }
    
    pub fn backward(&self, dout: f32) -> (f32, f32) {
        (dout * self.y, dout * self.x)
    }
}


pub struct AddLayer {}

impl AddLayer {
    pub fn new() -> Self {
        AddLayer {}
    }
    
    pub fn forward(&mut self, x: f32, y: f32) -> f32 {
        x + y
    }
    
    pub fn backward(&self, dout: f32) -> (f32, f32) {
        (dout, dout)
    }
}
