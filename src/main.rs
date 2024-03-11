use tch::{nn, Device, Tensor};

#[derive(Debug)]
struct LinearRegression {
    layer: nn::Linear,
}

impl LinearRegression {
    fn new(vs: &nn::Path, inDim: i64, outDim: i64) -> LinearRegression {
        let layer = nn::linear(vs, inDim, outDim, Default::default());
        LinearRegression { layer }
    }
}

impl nn::Module for LinearRegression {
    fn forward(&self, xs: &Tensor) -> Tensor {
        xs.view([-1, 1]).apply(&self.layer)
    }
}

fn main() {
    // Creating new variables for input and target 
    let input = Tensor::of_slice(&[1.0, 2.0, 3.0, 4.0, 5.0]).view([-1, 1]);
    let target = Tensor::of_slice(&[2.0, 4.0, 6.0, 8.0, 10.0]).view([-1, 1]);

    // Place device on GPU if available
    let device = Device::cuda_if_available();
    let vs = nn::Path::new_with_device("/linear_regression", device);

    // Setting model parameters
    let inDim = 1;
    let outDim = 1;
    let model = LinearRegression::new(&vs, inDim, outDim);

    // Setting the optimizer and training parameters
    let mut opt = nn::Adam::default().build(&vs, 1e-2)?;

    // Training model
    for epoch in 1..=1000 {
        let predicted = model.forward(&input);
        let loss = predicted.mse_loss(&target);

        opt.backward_step(&loss);

        if epoch % 100 == 0 {
            println!("Epoch: {:5}, Loss: {:?}", epoch, loss);
        }
    }

    // Testing the model with new data
    let testInput = Tensor::of_slice(&[6.0, 7.0, 8.0, 9.0, 10.0]).view([-1, 1]);
    let predictedOutput = model.forward(&testInput);

    println!("Test Input: {:?}", testInput);
    println!("Predicted Output: {:?}", predictedOutput);
}
