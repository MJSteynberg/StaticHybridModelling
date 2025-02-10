# Hybrid Modeling Project

This project explores hybrid modeling by combining synthetic and physical models. The goal is to create a framework that allows for the evaluation of unknown functions using synthetic models, while also incorporating physical models that solve partial differential equations (PDEs).

## Project Structure

```
my-hybrid-modeling
├── src
│   ├── main.py                # Entry point for the application
│   └── models
│       ├── __init__.py        # Initializes the models package
│       ├── synthetic_model.py  # Abstract class for synthetic models
│       └── physical_model.py   # Abstract class for physical models
├── requirements.txt            # Project dependencies
└── README.md                   # Project documentation
```

## Overview

The project consists of two main types of models:

1. **Synthetic Models**: These models do not rely on physical assumptions and can be any type of model that maps input space (x, y) to an output (u). Examples include Feedforward Neural Networks (FNN) and Residual Networks (ResNet).

2. **Physical Models**: These models are based on physical principles and solve equations like the Poisson equation using methods such as finite element analysis or finite difference methods. An example implementation is the PoissonFEM class.

## Getting Started

To set up the project, follow these steps:

1. Clone the repository:
   ```
   git clone <repository-url>
   ```

2. Navigate to the project directory:
   ```
   cd my-hybrid-modeling
   ```

3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Run the application:
   ```
   python src/main.py
   ```

## Future Work

This project will be expanded to include specific implementations of the synthetic and physical models, as well as the training process for minimizing loss on given data.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.