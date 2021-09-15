# QOSF-screening-task
QOSF task 2 for screening


Task 2


Prepare 4 random 4-qubit quantum states of your choice.
Create and train a variational circuit that transforms input states into predefined output states. Namely
if random state 1 is provided, it returns state |0011>
if random state 2 is provided, it returns state |0101>
if random state 3 is provided, it returns state |1010>
if random state 4 is provided, it returns state |1100>
What would happen if you provided a different state?

Analyze and discuss the results.

Feel free to use existing frameworks (e.g. PennyLane, Qiskit) for creating and training the circuits.
This PennyLane demo can be useful: Training a quantum circuit with Pytorch, 
This Quantum Tensorflow tutorial can be useful: Training a quantum circuit with Tensorflow .

For the variational circuit, you can try any circuit you want. You can start from one with a layer of RX, RY and CNOTs, repeated a couple of times (though there are certainly better circuits to achieve this goal). 

Context:
This challenge has been inspired by the following papers “A generative modeling approach for benchmarking and training shallow quantum circuits” and “Generation of High-Resolution Handwritten Digits with an Ion-Trap Quantum Computer“. The target states of this task can be interpreted as the 2x2 “bars and stripes” patterns used in the first paper.


# Files
* t2.py is the solution file.
* requirements.txt contains the required packages.
* Plots folder contains results obtained for layers -> 1 to 10
* accuracy_scores_comparison.png contains accuracy scores from all 10 cases
* fidelity_result_comparison.png contains fidelity from all 10 cases