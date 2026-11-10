import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, make_circles
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from matplotlib.colors import ListedColormap

st.set_page_config(page_title="Sigmoid Activation Function", layout="wide")
st.title("Sigmoid Activation Function")
st.write("Sigmoid transforms inputs into values between 0 and 1.")
st.latex(r"f(x) = \frac{1}{1+e^{-x}}")
st.latex(r"f'(x) = f(x)(1-f(x))")

# Sidebar settings
with st.sidebar:
    st.header("Activation Curve")
    x_input = st.slider("Input value x", -6.0, 6.0, 1.0, 0.1)
    st.header("Neural Network Demo")
    ds = st.selectbox("Dataset", ["Moons", "Circles"])
    n_samples = st.slider("Samples", 100, 1000, 300, 50)
    noise = st.slider("Noise", 0.0, 0.4, 0.2, 0.01)
    with st.expander("Advanced Settings"):
        hidden = st.text_input("Hidden layer sizes", "16,16")
        max_iter = st.slider("Max Iterations", 100, 3000, 800, 100)

# Activation function
def activation_fn(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def activation_derivative(x):
    s = activation_fn(x)
    return s * (1 - s)

# 1. Activation curve
st.subheader("1. Sigmoid Curve")
xgrid = np.linspace(-6, 6, 400)
output = activation_fn(x_input)
gradient = activation_derivative(x_input)

col1, col2 = st.columns(2)

with col1:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(xgrid, activation_fn(xgrid), color="green", label="Sigmoid")
    ax.plot(xgrid, activation_derivative(xgrid), "--",
            color="orange", label="Derivative")
    ax.scatter(x_input, output, color="red", label="Your input", zorder=5)
    ax.axhline(0, color="grey", linewidth=0.5)
    ax.axvline(0, color="grey", linewidth=0.5)
    ax.set(xlabel="Input (x)", ylabel="Output", title="Sigmoid Activation")
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    plt.close(fig)

with col2:
    st.write("**Output for your input**")
    a, b, c = st.columns(3)
    a.metric("Input", f"{x_input:.2f}")
    b.metric("Output", f"{output:.4f}")
    c.metric("Derivative", f"{gradient:.4f}")
    st.write("**Properties**")
    st.write("- Range: (0, 1)")
    st.write("- Smooth S-shaped curve")
    st.write("- Useful for binary classification outputs")
    st.write("- May experience vanishing gradients")

# 2. Neural network demonstration
st.subheader("2. Neural Network using Sigmoid")

if ds == "Moons":
    X, y = make_moons(n_samples=n_samples, noise=noise, random_state=42)
else:
    X, y = make_circles(n_samples=n_samples, noise=noise,
                        factor=0.5, random_state=42)

try:
    hidden_sizes = tuple(int(v.strip()) for v in hidden.split(","))
    if not hidden_sizes or any(v <= 0 for v in hidden_sizes):
        raise ValueError
except ValueError:
    st.error("Enter hidden layer sizes such as 16,16.")
    st.stop()

model = Pipeline([
    ("scaler", StandardScaler()),
    ("mlp", MLPClassifier(hidden_layer_sizes=hidden_sizes,
                          activation="logistic",
                          learning_rate_init=0.01,
                          max_iter=max_iter, random_state=42))
])
model.fit(X, y)
accuracy = model.score(X, y)

# Decision boundary
xx, yy = np.meshgrid(
    np.linspace(X[:, 0].min()-1, X[:, 0].max()+1, 250),
    np.linspace(X[:, 1].min()-1, X[:, 1].max()+1, 250)
)
Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

fig2, ax2 = plt.subplots(figsize=(7, 5))
ax2.contourf(xx, yy, Z, cmap=ListedColormap(["#FFEEEE", "#EEFFEE"]))
ax2.scatter(X[:, 0], X[:, 1], c=y, s=25, edgecolor="black",
            cmap=ListedColormap(["red", "blue"]))
ax2.set_title(f"MLP with Sigmoid (Training Accuracy = {accuracy:.3f})")
ax2.set(xlabel="x1", ylabel="x2")

left, middle, right = st.columns([1, 3, 1])
with middle:
    st.pyplot(fig2)
plt.close(fig2)

