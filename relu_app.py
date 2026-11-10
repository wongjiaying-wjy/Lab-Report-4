import warnings
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, make_circles
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.exceptions import ConvergenceWarning
from matplotlib.colors import ListedColormap

st.set_page_config(page_title="Rectified Linear Unit (ReLU)", page_icon="🧠", layout="wide")
st.title("Rectified Linear Unit (ReLU)")
st.caption("BSD3513 Introduction to Artificial Intelligence · Lab Report 4 · SD24048 WONG JIA YING")
st.write("ReLU returns zero for negative inputs and the input itself for positive inputs.")
st.latex(r"f(x) = \max(0,x)")
st.latex(r"f'(x) = \begin{cases}0 & x < 0\\1 & x > 0\end{cases}")

with st.sidebar:
    st.header("Activation Curve")
    x_input = st.slider("Input value x", -6.0, 6.0, 1.0, 0.1)

    st.header("Neural Network Demo")
    dataset = st.selectbox("Dataset", ["Moons", "Circles"])
    n_samples = st.slider("Samples", 100, 1000, 300, 50)
    noise = st.slider("Noise", 0.00, 0.40, 0.20, 0.01)
    with st.expander("Advanced settings"):
        hidden_text = st.text_input("Hidden layer sizes (e.g., 16,16)", "16,16")
        max_iter = st.slider("Maximum iterations", 100, 3000, 800, 100)
    st.caption("Random seed: 42 (fixed for reproducibility)")

RANDOM_STATE = 42

def relu(x):
    return np.maximum(0, x)


def relu_derivative(x):
    return np.where(x > 0, 1.0, 0.0)

st.subheader("1. Relu Curve")
x_values = np.linspace(-6, 6, 400)
col_curve, col_output = st.columns(2)

with col_curve:
    with st.container(border=True):
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(x_values, relu(x_values), label="Relu f(x)", color="royalblue", linewidth=2)
        ax.plot(x_values, relu_derivative(x_values), label="Derivative f'(x)", color="darkorange", linestyle="--")
        ax.scatter([x_input], [relu(x_input)], color="crimson", s=70, zorder=5, label="Selected input")
        ax.axhline(0, color="black", linewidth=0.6)
        ax.axvline(0, color="black", linewidth=0.6)
        ax.set(xlabel="Input x", ylabel="Output", title="Relu Activation")
        ax.grid(True, alpha=0.3)
        ax.legend()
        st.pyplot(fig)
        plt.close(fig)

with col_output:
    with st.container(border=True):
        st.markdown("**Output for selected input**")
        m1, m2, m3 = st.columns(3)
        m1.metric("x", f"{x_input:.2f}")
        m2.metric("Relu(x)", f"{float(relu(x_input)):.4f}")
        m3.metric("f'(x)", f"{float(relu_derivative(x_input)):.4f}")
        st.markdown("**Properties**")
        st.write("• Range: [0, ∞)")
        st.write("• Simple and computationally efficient")
        st.write("• Positive inputs retain a gradient of 1")
        st.write("• Negative inputs have zero gradient (dying ReLU risk)")
        st.caption("The derivative is undefined at x = 0; this demo uses 0 by convention.")

st.subheader("2. Neural Network using Relu")
try:
    hidden_sizes = tuple(int(part.strip()) for part in hidden_text.split(","))
    if not hidden_sizes or any(size < 1 or size > 256 for size in hidden_sizes) or len(hidden_sizes) > 5:
        raise ValueError
except ValueError:
    st.error("Enter 1–5 positive hidden-layer sizes (1–256), separated by commas, e.g., 16,16.")
    st.stop()

if dataset == "Moons":
    X, y = make_moons(n_samples=n_samples, noise=noise, random_state=RANDOM_STATE)
else:
    X, y = make_circles(n_samples=n_samples, noise=noise, factor=0.5, random_state=RANDOM_STATE)

model = Pipeline([
    ("scaler", StandardScaler()),
    ("mlp", MLPClassifier(
        hidden_layer_sizes=hidden_sizes,
        activation="relu",
        learning_rate_init=0.01,
        max_iter=max_iter,
        random_state=RANDOM_STATE,
    )),
])

with st.spinner("Training neural network..."):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(X, y)
    training_accuracy = model.score(X, y)

st.metric("Training accuracy", f"{training_accuracy * 100:.2f}%")
st.caption("Training accuracy is measured on the same data used to fit the model; it is not a test-set score.")
if any(issubclass(w.category, ConvergenceWarning) for w in caught):
    st.warning("The model reached its stopping condition without meeting the convergence tolerance. Try increasing maximum iterations.")

x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 300),
    np.linspace(y_min, y_max, 300),
)
grid = np.c_[xx.ravel(), yy.ravel()]
predicted = model.predict(grid).reshape(xx.shape)

fig_boundary, ax_boundary = plt.subplots(figsize=(7, 5))
ax_boundary.contourf(xx, yy, predicted, cmap=ListedColormap(["#FFEEEE", "#EEFFEE"]), alpha=0.75)
ax_boundary.scatter(
    X[:, 0], X[:, 1], c=y, cmap=ListedColormap(["#CC0000", "#007700"]),
    s=24, edgecolors="black", linewidths=0.35,
)
ax_boundary.set_title("MLP with Relu — Training accuracy: " + f"{training_accuracy:.3f}")
ax_boundary.set_xlabel("x1")
ax_boundary.set_ylabel("x2")
ax_boundary.grid(True, alpha=0.15)

left, center, right = st.columns([1, 3, 1])
with center:
    st.pyplot(fig_boundary)
plt.close(fig_boundary)

with st.expander("Interpretation", expanded=False):
    st.write("ReLU returns zero for negative inputs and the input itself for positive inputs.")
    st.write("The coloured background shows the classes predicted by the trained MLP. "
             "The plotted points show the actual dataset labels. The boundary changes "
             "when the dataset, noise, or network settings change.")
    st.caption("This demonstration uses a fixed random seed for reproducibility.")
