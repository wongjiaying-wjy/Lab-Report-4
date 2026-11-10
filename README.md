
# 🧠 Activation Function Visualiser: ReLU, Sigmoid and Tanh

**BSD3513 Introduction to Artificial Intelligence · Lab Report 4**

**SD24048 WONG JIA YING**

🌐 **Live Applications:**

- **ReLU:** [Add ReLU Streamlit URL]
- **Sigmoid:** [Add Sigmoid Streamlit URL]
- **Tanh:** [Add Tanh Streamlit URL]

💻 **Source Code:** [Add GitHub Repository URL]

## About

Three interactive Streamlit applications that visualise the behaviour of neural network activation functions: **Rectified Linear Unit (ReLU), Sigmoid, and Hyperbolic Tangent (Tanh)**.

Each application displays the activation function and its derivative, allowing users to adjust the input value and observe the corresponding outputs.

The applications also demonstrate how each activation function is used in a **Multilayer Perceptron (MLP)** for nonlinear classification using the Moons or Circles dataset. The trained model's decision boundary and training accuracy are displayed to illustrate its classification performance.

## Activation Functions

| Function | Formula | Derivative | Output Range |
|---|---|---|---|
| ReLU | max(0, x) | 1 if x > 0, otherwise 0 | [0, ∞) |
| Sigmoid | 1 / (1 + e^(-x)) | f(x)(1 - f(x)) | (0, 1) |
| Tanh | (e^x - e^(-x)) / (e^x + e^(-x)) | 1 - tanh²(x) | (-1, 1) |

**Adjustable parameters**

| Parameter | Default | Range / Options |
|---|---|---|
| Input value (x) | 1.0 | -6.0 to 6.0 |
| Dataset | Moons | Moons, Circles |
| Number of samples | 300 | 100 to 1,000 |
| Noise level | 0.20 | 0.00 to 0.40 |
| Hidden layer sizes | 16,16 | Customisable |
| Maximum iterations | 800 | 100 to 3,000 |
| Random seed | 42 | Fixed |

The random seed is fixed at 42 to ensure reproducible results under the same settings.

## Application Features

### 1. ReLU Activation Function

The ReLU application displays the activation curve and its derivative for input values between -6 and 6. ReLU produces zero for negative inputs and returns the input value for positive inputs.

Users can adjust the input slider to observe the activation output and derivative. The application also displays the key properties of ReLU, including its computational simplicity and the possibility of inactive neurons.

A neural network using ReLU is trained on the selected dataset, and its decision boundary illustrates how the model separates nonlinear data.

![ReLU activation function](image/relu.png)

### 2. Sigmoid Activation Function

The Sigmoid application visualises the S-shaped activation curve and its derivative. Sigmoid transforms input values into a range between 0 and 1.

Users can adjust the input slider to examine how the output changes. The application also explains the function's properties, including its suitability for binary classification outputs and the vanishing gradient problem.

A neural network using Sigmoid is trained on the selected dataset, with its decision boundary and training accuracy displayed.

![Sigmoid activation function](image/sigmoid.png)

### 3. Tanh Activation Function

The Tanh application displays the activation curve and its derivative. Tanh transforms input values into a range between -1 and 1, producing zero-centred outputs.

Users can adjust the input slider to observe the activation output and derivative. The application highlights its zero-centred behaviour and the possibility of vanishing gradients for large input values.

A neural network using Tanh is trained on the selected dataset to demonstrate nonlinear classification through its decision boundary and training accuracy.

![Tanh activation function](image/tanh.png)

### 4. Neural Network Decision Boundary

Each application trains a Multilayer Perceptron (MLP) using the selected activation function.

The **Moons** and **Circles** datasets provide nonlinear classification problems. The decision boundary visualises the regions assigned to each class, while the data points represent the actual class labels.

The displayed training accuracy indicates the proportion of training samples correctly classified by the model.

## Interpretation of Results

### ReLU

The ReLU curve remains at zero for negative inputs and increases linearly for positive inputs. Its derivative is zero for negative inputs and one for positive inputs, allowing gradients to propagate effectively in the positive region.

The neural network demonstrates that ReLU can learn nonlinear decision boundaries. However, neurons with consistently negative inputs may stop updating because their gradients become zero.

### Sigmoid

The Sigmoid curve increases smoothly from approximately zero to one. Its derivative reaches a maximum of 0.25 at x = 0 and approaches zero for large positive or negative inputs.

The neural network can learn nonlinear classification patterns using Sigmoid. However, small gradients in saturated regions may slow down training due to the vanishing gradient problem.

### Tanh

The Tanh curve increases smoothly from -1 to 1 and is centred at zero. Its derivative reaches a maximum of 1 at x = 0 and approaches zero for large absolute input values.

The zero-centred output can support more balanced learning compared with Sigmoid, although Tanh may still experience vanishing gradients in saturated regions.

### Conclusion

The three applications successfully demonstrate the characteristics of ReLU, Sigmoid, and Tanh activation functions through interactive curves, derivatives, and neural network decision boundaries.

ReLU provides non-saturating gradients for positive inputs, Sigmoid produces outputs between 0 and 1, and Tanh produces zero-centred outputs between -1 and 1.

The results illustrate how different activation functions influence neural network behaviour and nonlinear classification.

## How to Run

Install the required libraries:

```bash
pip install -r requirements.txt
```

Run each application separately:

**ReLU**
```bash
streamlit run relu_app.py
```

**Sigmoid**
```bash
streamlit run sigmoid_app.py
```

**Tanh**
```bash
streamlit run tanh_app.py
```

The selected application usually opens at http://localhost:8501.

## How the Neural Network Works

1. **Dataset generation:** The application generates a Moons or Circles dataset with the selected number of samples and noise level.
2. **Data preprocessing:** StandardScaler standardises the input features before model training.
3. **Network configuration:** An MLP is constructed using the selected activation function and hidden layer sizes.
4. **Forward propagation:** Input data passes through the network to generate predictions.
5. **Model training:** The MLP updates its weights and biases to minimise prediction errors.
6. **Classification:** The trained model predicts the class of each input sample.
7. **Decision boundary:** The application visualises the classification regions and original data points.
8. **Performance evaluation:** Training accuracy is calculated to measure how well the model classifies the training samples.

## Repository Structure

```text
Lab-Report-4/
├── relu_app.py
├── sigmoid_app.py
├── tanh_app.py
├── requirements.txt
├── README.md
└── image/
    ├── relu.png
    ├── sigmoid.png
    └── tanh.png
```

## Built with

Python · Streamlit · NumPy · Matplotlib · scikit-learn

## Complexity Note

Each application generates a dataset containing N samples and trains a Multilayer Perceptron using the selected network architecture.

The computational cost of training depends on the number of samples, network connections, and training iterations. For a fixed network architecture, the training cost increases approximately with the number of samples and training iterations.

The decision boundary is generated using a 300 × 300 grid, requiring 90,000 predictions to visualise the classification regions.

The random seed is fixed at 42, allowing the same dataset and model settings to produce reproducible results.
