import matplotlib.pyplot as plt
import pennylane as qml
import pennylane.numpy as np
import seaborn as sns

sns.set_theme()
rng = np.random.default_rng(42)

n_qubits = 4
possible_states = 2 ** n_qubits

random_states = rng.choice(
    possible_states, 4, replace=False
)  # Select any 4 random states. These will form our input data.

label_states = [3, 5, 10, 12]  # Label states from the question
data_points = 100  # Number of points in the dataset.

ini_data = [(i, j) for i, j in zip(random_states, label_states)]
input_data = rng.choice(
    ini_data, data_points, replace=True
)  # Generate input data. Contains copies of the random states generated above.

x = np.array([i[0] for i in input_data])  # 'Features'
y = np.array([i[1] for i in input_data])  # 'Labels'

dev = qml.device("default.qubit", wires=n_qubits)
n_layers = 2  # Change number of layers here.

params = rng.random([n_qubits, n_layers, 3])  # Random parameter initialization


@qml.qnode(dev)
def circuit_prep(n):
    """
    Create the circuit to prepare state |x> where x is the binary representation of n
    :param n: The number n to be converted into a quantum state
    :return: The state |x>. x -> binary representation of n
    """
    n = "{0:b}".format(int(n)).zfill(n_qubits)
    for i in range(n_qubits):
        if n[i] == "1":
            qml.PauliX(wires=i)
    return qml.state()


def state_prep(n):
    """
    Return the state |x>. x -> binary representation of n
    """
    return circuit_prep(n)


@qml.qnode(dev)
def var_circ(parameters, x_=None):
    """
    Variational circuit. Map input states to fixed output states.
    :param parameters: Training parameters
    :param x_: Input state to be mapped
    :return: Mapped output state
    """
    n = "{0:b}".format(int(x_)).zfill(n_qubits)
    for i in range(n_qubits):
        if n[i] == "1":
            qml.PauliX(wires=i)
    for i in range(n_layers):
        for j in range(n_qubits):
            qml.Rot(*parameters[j][i], wires=j)
        qml.broadcast(qml.CNOT, wires=range(n_qubits), pattern="ring")
    return qml.state()


def cost_func(parameters, x_, y_):
    """
    Fidelity cost function. Calculates state fidelity between output and label states.
    1 is similar.
    0 is orthogonal.
    :param parameters: Training parameters
    :param x_: Input states
    :param y_: Label states
    :return: Negative of average fidelity. (-ve as minimising this will maximize the fidelity.)
    """
    tot = 0
    for i, j in zip(x_, y_):
        fid = np.abs(np.dot(np.conj(var_circ(parameters, i)), state_prep(j))) ** 2
        tot = tot + fid
    return -tot / len(x_)


def predict(x_):
    """
    Function to get mapped states from input states.
    :param x_: Input state
    :return: Output state
    """
    fids = [np.abs(np.dot(np.conj(var_circ(params, x_)), state_prep(j))) ** 2 for j in label_states]
    return label_states[np.asarray(fids).argmax()]


def score(x_, y_):
    """
    Calculate accuracy score.
    :param x_: Input states
    :param y_: Label states
    :return: Accuracy score
    """
    labels = np.asarray([predict(i) for i in x_])
    return sum(labels == y_) / len(x_)


opt = qml.AdamOptimizer()  # Optimizer
batch_size = int(len(x) * 0.1)  # Batch size of data to be optimized in each step.
costs = [cost_func(params, x, y)]  # Store fidelity for each step
scores = [score(x, y)]  # Store accuracy for each step

for i in range(1, 250):
    batch = rng.integers(0, len(x), size=(batch_size,))
    X_batch = x[batch]
    y_batch = y[batch]
    params = opt.step(lambda w: cost_func(w, X_batch, y_batch), params)  # Optimization step
    costs.append(cost_func(params, x, y))
    scores.append(score(x, y))

# Plot results
plt.plot(range(len(costs)), -np.asarray(costs))
plt.ylabel("State Fidelity")
plt.xlabel("No. of steps")
plt.xlim(left=0)
plt.ylim([0, 1])
plt.title(f"State Fidelity with {n_layers} layers")
plt.tight_layout()
plt.show()

plt.plot(range(len(scores)), scores, ":*")
plt.ylabel("Accuracy")
plt.xlabel("No. of steps")
plt.xlim(left=0)
plt.ylim([0, 1])
plt.title(f"Accuracy score with {n_layers} layers")
plt.tight_layout()
plt.show()
