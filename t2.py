import time

import matplotlib.pyplot as plt
import pennylane as qml
import pennylane.numpy as np

n_qubits = 4

possible_states = 2 ** 4

rng = np.random.default_rng(42)

random_states = rng.choice(possible_states, 4, replace=False)
print(random_states)

label_states = [3, 5, 10, 12]
data_points = 100

ini_data = [(i, j) for i, j in zip(random_states, label_states)]
input_data = rng.choice(ini_data, data_points, replace=True)

x = np.array([i[0] for i in input_data])
y = np.array([i[1] for i in input_data])

dev = qml.device('default.qubit', wires=n_qubits)
n_layers = 1
params = rng.random([n_qubits, n_layers, 3])


@qml.qnode(dev)
def circuit_prep(n):
    n = "{0:b}".format(int(n)).zfill(n_qubits)
    for i in range(n_qubits):
        if n[i] == '1':
            qml.PauliX(wires=i)
    return qml.state()


def state_prep(n):
    return circuit_prep(n)


@qml.qnode(dev)
def var_circ(params, x=None):
    n = "{0:b}".format(int(x)).zfill(n_qubits)
    for i in range(n_qubits):
        if n[i] == '1':
            qml.PauliX(wires=i)
    for i in range(n_layers):
        for j in range(n_qubits):
            qml.Rot(*params[j][i], wires=j)
        qml.broadcast(qml.CNOT, wires=range(n_qubits), pattern='ring')
    return qml.state()


def cost_func(params, x, y):
    tot = 0
    for i, j in zip(x, y):
        fid = np.abs(np.dot(np.conj(var_circ(params, i)), state_prep(j))) ** 2
        tot = tot + fid
    return -tot / len(x)


def predict(x):
    fids = [np.abs(np.dot(np.conj(var_circ(params, x)), state_prep(j))) ** 2 for j in label_states]
    return label_states[np.asarray(fids).argmax()]


def score(x, y):
    labels = np.asarray([predict(i) for i in x])
    return sum(labels == y) / len(x)


opt = qml.AdamOptimizer()
batch_size = int(len(x) * .1)
costs = [cost_func(params, x, y)]
t1 = time.time()
scores = []
for i in range(1, 100):
    batch = rng.integers(0, len(x), size=(batch_size,))
    X_batch = x[batch]
    y_batch = y[batch]
    params = opt.step(lambda w: cost_func(w, X_batch, y_batch), params)
    costs.append(cost_func(params, x, y))
    scores.append(score(x, y))
    if i % 10 == 0:
        print(i)

plt.plot(range(len(costs)), costs)
plt.title(f'costs {n_layers}')
plt.show()

plt.plot(range(len(scores)), scores)
plt.title(f'scores {n_layers}')
plt.show()
