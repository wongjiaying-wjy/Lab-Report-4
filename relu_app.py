import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.datasets import make_moons, make_circles
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning
import warnings

# 1. Page setup
st.set_page_config(page_title='Rectified Linear Unit (ReLU)', layout="wide")
st.title('Rectified Linear Unit (ReLU)')
st.write('ReLU returns the input if it is positive; otherwise, it returns zero.')
st.latex('f(x) = \\max(0,x)')
st.latex("f'(x) = \\begin{cases}1 & x > 0 \\\\ 0 & x \\leq 0\\end{cases}")

# 2. Sidebar controls
with st.sidebar:
    st.header("Activation Curve")
    x_input = st.slider("Input value x", -6.0, 6.0, 1.0, 0.1)

    st.header("Neural Network Demo")
    dataset = st.selectbox("Dataset", ["Moons", "Circles"])
    n_samples = st.slider("Samples", 100, 1000, 300, 50)
    noise = st.slider("Noise", 0.0, 0.4, 0.2, 0.01)
    with st.expander("Advanced settings"):
        hidden = st.text_input("Hidden layer sizes (e.g. 16,16)", "16,16")
        max_iter = st.slider("Max Iterations", 100, 3000, 800, 100)

RANDOM_STATE = 42

# 3. Activation function and derivative
def activation_fn(x):
    return np.maximum(0, x)


def activation_derivative(x):
    return np.where(np.asarray(x) > 0, 1.0, 0.0)

# 4. Interactive activation curve
st.subheader("1. ReLU Curve")
x_values = np.linspace(-6, 6, 400)
output = float(activation_fn(x_input))
gradient = float(activation_derivative(x_input))

left, right = st.columns(2)
with left:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(x_values, activation_fn(x_values), label="Activation f(x)", linewidth=2)
    ax.plot(x_values, activation_derivative(x_values), "--", label="Derivative f'(x)")
    ax.scatter(x_input, output, color="red", zorder=5, label="Selected input")
    ax.axhline(0, color="grey", linewidth=0.6)
    ax.axvline(0, color="grey", linewidth=0.6)
    ax.set_xlabel("Input (x)")
    ax.set_ylabel("Output")
    ax.grid(alpha=0.3)
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)

with right:
    st.write("**Output for your input**")
    a, b, d = st.columns(3)
    a.metric("Input", f"{x_input:.2f}")
    b.metric("Output", f"{output:.4f}")
    d.metric("Derivative", f"{gradient:.4f}")
    st.write("**Properties**")
    st.write('- Range: [0, ∞)')
    st.write('- Simple and fast to compute')
    st.write('- Keeps a useful gradient for positive inputs')
    st.write('- Negative inputs can lead to dying ReLU')
    st.caption('At x = 0, the ReLU derivative is undefined; this app uses 0 by convention.')

# 5. Dataset and model
st.subheader("2. Neural Network using ReLU")
try:
    hidden_sizes = tuple(int(value.strip()) for value in hidden.split(","))
    if not hidden_sizes or any(size <= 0 for size in hidden_sizes):
        raise ValueError
except ValueError:
    st.error("Hidden layer sizes must be positive integers separated by commas, e.g. 16,16.")
    st.stop()

if dataset == "Moons":
    X, y = make_moons(n_samples=n_samples, noise=noise, random_state=RANDOM_STATE)
else:
    X, y = make_circles(n_samples=n_samples, noise=noise, factor=0.5,
                        random_state=RANDOM_STATE)

model = Pipeline([
    ("scaler", StandardScaler()),
    ("mlp", MLPClassifier(hidden_layer_sizes=hidden_sizes,
                          activation='relu',
                          learning_rate_init=0.01,
                          max_iter=max_iter,
                          random_state=RANDOM_STATE))
])

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always", ConvergenceWarning)
    model.fit(X, y)
if any(issubclass(w.category, ConvergenceWarning) for w in caught):
    st.warning("Training reached the iteration limit before convergence. Try increasing Max Iterations.")

accuracy = model.score(X, y)  # Accuracy on training data, not unseen test data

# 6. Decision boundary
xx, yy = np.meshgrid(
    np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 250),
    np.linspace(X[:, 1].min() - 1, X[:, 1].max() + 1, 250)
)
Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

fig2, ax2 = plt.subplots(figsize=(7, 5))
ax2.contourf(xx, yy, Z, cmap=ListedColormap(["#FFEEEE", "#EEFFEE"]), alpha=0.7)
ax2.scatter(X[:, 0], X[:, 1], c=y, s=24, edgecolor="black",
            linewidth=0.4, cmap=ListedColormap(["#CC0000", "#0066AA"]))
ax2.set_title(f"MLP with ReLU (Training Accuracy = {accuracy:.3f})")
ax2.set_xlabel("x1")
ax2.set_ylabel("x2")

_, middle, _ = st.columns([1, 3, 1])
with middle:
    st.pyplot(fig2)
plt.close(fig2)

st.caption("Training accuracy is calculated using the same samples used to fit the model.")
with st.expander("Interpretation"):
    st.write('ReLU is zero for negative inputs and linear for positive inputs. Its positive-side gradient helps learning, but consistently negative inputs can cause inactive neurons.')
    st.write("The coloured background shows the classes predicted by the MLP. "
             "The dots are the actual data points. A boundary that follows the "
             "dataset's shape generally indicates a better fit to these samples.")
