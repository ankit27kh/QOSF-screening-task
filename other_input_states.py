import pickle
import pandas as pd
from matplotlib import pyplot as plt

with open('10param.pkl', 'rb') as f:
    params = pickle.load(f)

import pennylane as qml
import pennylane.numpy as np

rng = np.random.default_rng(42)

n_qubits = 4
possible_states = 2 ** n_qubits

"""
7 -> 3
1 -> 5
9 -> 10
10 -> 12
"""

label_states = [i for i in range(16)]
x = [i for i in range(16)]

dev = qml.device("default.qubit", wires=n_qubits)
n_layers = 10


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


def predict(x_):
    """
    Function to get mapped states from input states.
    :param x_: Input state
    :return: Output state
    """
    fids = [np.abs(np.dot(np.conj(var_circ(params, x_)), state_prep(j))) ** 2 for j in label_states]
    return label_states[np.asarray(fids).argmax()], max(fids), fids


fidelities = []
max_fidelities = []
labels = []
for i in x:
    res = predict(i)
    labels.append(res[0])
    max_fidelities.append(round(res[1], 2))
    fidelities.append([round(j, 2) for j in res[2]])

results = {}
res = {}
for i, j, l in zip(x, labels, fidelities):
    results[i] = j
    res[i] = l

res = pd.DataFrame(res)
res.T.plot.bar(legend=None)
plt.xlabel("Input States")
plt.ylabel("Fidelity")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

results = pd.DataFrame([results], index=['Output State'])
print(results)
