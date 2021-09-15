# QOSF-screening-task
QOSF task 2 for screening


### Task 2


* Prepare 4 random 4-qubit quantum states of your choice.
* Create and train a variational circuit that transforms input states into predefined output states. Namely
  * if random state 1 is provided, it returns state |0011>
  * if random state 2 is provided, it returns state |0101>
  * if random state 3 is provided, it returns state |1010>
  * if random state 4 is provided, it returns state |1100>
* What would happen if you provided a different state?

Analyze and discuss the results.

Feel free to use existing frameworks (e.g. PennyLane, Qiskit) for creating and training the circuits.
This PennyLane demo can be useful: [Training a quantum circuit with Pytorch](https://pennylane.ai/qml/demos/tutorial_state_preparation.html)
This Quantum Tensorflow tutorial can be useful: [Training a quantum circuit with Tensorflow](https://www.tensorflow.org/quantum/tutorials/mnist)

For the variational circuit, you can try any circuit you want. You can start from one with a layer of RX, RY and CNOTs, repeated a couple of times (though there are certainly better circuits to achieve this goal). 

### Context:
This challenge has been inspired by the following papers [A generative modeling approach for benchmarking and training shallow quantum circuits](https://www.nature.com/articles/s41534-019-0157-8) and [Generation of High-Resolution Handwritten Digits with an Ion-Trap Quantum Computer](https://arxiv.org/abs/2012.03924). The target states of this task can be interpreted as the 2x2 “bars and stripes” patterns used in the first paper.


# Files
* t2.py is the solution file.
* requirements.txt contains the required packages.
* Plots folder contains results obtained for layers -> 1 to 10
* accuracy_scores_comparison.png contains accuracy scores from all 10 cases
* fidelity_result_comparison.png contains fidelity from all 10 cases


# Description
First step is to get 4 random states. These will act as our input data.
We create 100 data points by repeating these states. Each state has a label state as given in the question.

The label states and input states are represented by integers at this point. We have one circuit to map the label states to their binary representation states. These will be used to measure state fidelity.

The training circuit is simple. It contains arbitrary single qubit rotation on each qubit and a ring of CNOT gates covering all qubits in each layer.
For each data point (integer), X gates are placed to make the input state of the circuit as the binary represented state of the data point. Then the variational circuit is applied on this input state.
The output state is compared with the labels by calculating the state fidelity.

Accuracy of the circuit is tested by checking if our output state corresponds to the label state. This is done by checking the fidelity of the output state with all 4 label states and assigning the max fidelity state.


# Results
From the plots we can see that with increasing layers we achieve 1 state fidelity. This means output states are completely same to the label states.
Accuracy reaches 1 much earlier than fidelity. We can explain this by noting that we are assigning max fidelity state out of the 4 labels to the data point. Thus, we don't need a perfect match.
We can also note that for fewer layers, the fidelity reaches a steady state and is unlikely to improve with higher optimization steps.
