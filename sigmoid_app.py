import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from matplotlib.colors import ListedColormap
 
st.set_page_config(page_title="Sigmoid Activation Function", layout="wide")
st.title("Sigmoid Activation Function")
st.write(
    """
    The Sigmoid (logistic) function squashes any input into the range (0, 1), so it is often used for probabilities in binary classification output layers.
    """
)
 
# --- Activation function and its derivative ---
def activation(x):
    return 1 / (1 + np.exp(-x))
 
 
def derivative(x):
    s = 1 / (1 + np.exp(-np.asarray(x)))
    return s * (1 - s)
 
 
# --- Sidebar controls ---
with st.sidebar:
    st.header("Function Range")
    x_min, x_max = st.slider("x range", -10.0, 10.0, (-6.0, 6.0), 0.5)
    n_points = st.slider("Number of points", 50, 1000, 400, 50)
    show_derivative = st.checkbox("Show derivative (gradient)", value=True)
 
    st.header("Single Neuron")
    w = st.slider("Weight (w)", -3.0, 3.0, 1.0, 0.1)
    b = st.slider("Bias (b)", -3.0, 3.0, 0.0, 0.1)
 
    st.header("MLP Demo (logistic)")
    n_samples = st.slider("Samples", 100, 1000, 300, 50)
    noise = st.slider("Noise", 0.00, 0.40, 0.20, 0.01)
    hidden = st.text_input("Hidden layer sizes (e.g., 16,16)", value="16,16")
    max_iter = st.slider("Max iterations", 100, 2000, 600, 100)
    seed = st.number_input("Random seed", 0, 9999, 42)
 
# --- 1. Function curve ---
st.subheader("1. Activation Function Curve")
x = np.linspace(x_min, x_max, n_points)
 
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(x, activation(x), label="Sigmoid(x)", color="#1f77b4", linewidth=2)
if show_derivative:
    ax.plot(x, derivative(x), label="Derivative", color="#d62728", linestyle="--")
ax.axhline(0, color="grey", linewidth=0.8)
ax.axvline(0, color="grey", linewidth=0.8)
ax.set_xlabel("x"); ax.set_ylabel("f(x)")
ax.set_title("Sigmoid and its derivative")
ax.grid(True); ax.legend()
st.pyplot(fig)
 
st.markdown("**Formula:** f(x) = 1 / (1 + e^(-x))")
st.markdown("**Output range:** (0, 1)")
 
# --- 2. Single neuron ---
st.subheader("2. Single Neuron: f(w * x + b)")
z = w * x + b
fig2, ax2 = plt.subplots(figsize=(7, 4))
ax2.plot(x, z, label="z = w*x + b", color="grey", linestyle=":")
ax2.plot(x, activation(z), label="f(z)", color="#2ca02c", linewidth=2)
ax2.set_xlabel("x"); ax2.set_ylabel("Output")
ax2.set_title(f"Neuron output (w={w:.1f}, b={b:.1f})")
ax2.grid(True); ax2.legend()
st.pyplot(fig2)
 
x_in = st.number_input("Try an input value x", value=1.0, step=0.5)
z_in = w * x_in + b
st.write(f"z = w*x + b = **{z_in:.3f}**  →  f(z) = **{float(activation(z_in)):.3f}**, "
         f"gradient = **{float(derivative(z_in)):.3f}**")
 
# --- 3. MLP using this activation ---
st.subheader("3. Decision Boundary of an MLP using Sigmoid")
X, y = make_moons(n_samples=n_samples, noise=noise, random_state=int(seed))
 
hidden_sizes = tuple(int(s.strip()) for s in hidden.split(",") if s.strip().isdigit())
if not hidden_sizes:
    hidden_sizes = (16, 16)
 
clf = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", MLPClassifier(hidden_layer_sizes=hidden_sizes,
                          activation="logistic",
                          learning_rate_init=0.01,
                          max_iter=max_iter,
                          random_state=int(seed))),
])
clf.fit(X, y)
score = clf.score(X, y)
 
xx, yy = np.meshgrid(
    np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 300),
    np.linspace(X[:, 1].min() - 1, X[:, 1].max() + 1, 300),
)
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
 
fig3, ax3 = plt.subplots(figsize=(6, 5))
ax3.contourf(xx, yy, Z, cmap=ListedColormap(["#FFEEEE", "#EEFFEE"]), alpha=0.7)
ax3.scatter(X[:, 0], X[:, 1], c=y, s=25, edgecolor="k",
            cmap=ListedColormap(["#CC0000", "#007700"]))
ax3.set_title(f"MLP with Sigmoid (train acc={score:.3f})")
ax3.set_xlabel("x1"); ax3.set_ylabel("x2")
st.pyplot(fig3)
 
# --- Notes ---
st.subheader("Key Points")
st.markdown("""
- Smooth and differentiable, output interpretable as a probability.
- Saturates for large |x|, so gradients become very small (vanishing gradient).
- Output is not zero-centred; its maximum gradient is only 0.25.
""")
