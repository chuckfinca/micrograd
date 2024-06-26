import random
from enum import Enum

import numpy
from graphviz import Digraph

# Setup README.md generation
import os
import readme_ation

readme_path = 'README.md'
project_details = {
    'overview': "This project involves building a neural network from scratch inspired by Andrej Karpathy's video [\"The spelled-out intro to neural networks and backpropagation: building micrograd\"](https://www.youtube.com/watch?v=VMj-3S1tku0). The primary goal was to deepen my understanding of neural networks, backpropagation, and gradient descent by implementing these concepts manually.",
    'motivation': "Prior to pursuing my master's in ML/AI, my knowledge was primarily derived from several Coursera courses on machine learning and readings from Twitter. I sought a practical project to solidify this theoretical foundation. Karpathy's tutorial provided an excellent starting point due to his expertise and clear explanations.",
    'technologies': "- **Python**: Core programming language.\n- **NumPy**: For numerical operations and data handling.\n- **Graphviz**: For visualizing neural networks, weights, and gradients.",
    'approach': "1. **Learning by Doing**: Followed Karpathy’s tutorial, then independently implemented and expanded upon his concepts.\n2. **Extended Operations**: Added additional operations such as `tanh` to the neural network.\n3. **Commenting and Documentation**: Emphasized writing clear comments to aid future understanding and reinforce learning.\n4. **Custom Neuron Objects**: Implemented a `Neuron` class to simulate action potentials, though later realized `ReLU` serves this purpose in traditional neural networks.",
    'challenges': "- **Labeling System**: Attempted to implement a comprehensive labeling system, which proved impractical for larger networks.\n- **Understanding ReLU**: Initially misunderstood the role of `ReLU`, leading to an unnecessary `Neuron` class.\n- **Practical Application**: Applied the network to a Kaggle competition task, achieving 55% accuracy (basically chance), highlighting the need for further refinement and understanding.",
    'key_takeaways': "- **Hands-On Experience**: Building a neural network from scratch provided deep insights into the mechanics of neural networks.\n- **Importance of Iteration**: Recognized the importance of iterative learning and continuous improvement.\n- **Foundation for Future Projects**: The experience laid a solid foundation for more complex machine learning projects.",
    'acknowledgments': "Special thanks to Andrej Karpathy for his [invaluable tutorial](https://www.youtube.com/watch?v=VMj-3S1tku0), which guided the development of this project."
}
readme_ation.add_project_description(readme_path, project_details)

script_path = os.path.abspath(__file__)
readme_ation.add_setup_with_versions([script_path], readme_path)

def single_neuron_example():
    # NEURON EXAMPLE @ 56min

    # inputs x1, x2
    x1 = Value(2.0, label="x1")
    x2 = Value(0.0, label="x2")

    # weights w1, w2
    w1 = Value(-3.0, label="w1")
    w2 = Value(1.0, label="w=2")

    # bias of the neuron (trigger happiness)
    b = Value(6.881373587, label='b')

    # x1*w1 + x2*w2 + b

    x1w1 = x1 * w1
    x1w1.label = 'x1*w1'
    x2w2 = x2 * w2
    x2w2.label = 'x2*w2'
    x1w1x2w2 = x1w1 + x2w2
    # x1w1x2w2.label = 'x1w1 + x2w2'
    n = x1w1x2w2 + b
    n.label = 'n'

    o = n.tanh("o")

    o.back_propagation(False)

    draw_dot(o).view()

    torch_time()


def initial_example():
    # INITIAL EXAMPLE @ ~45min

    # inputs x1, x2
    a = Value(2.0, label="a")
    b = Value(-3.0, label="b")
    c = Value(10.0, label="c")
    e = a * b; e.label = "e"
    d = e + c; d.label = "d"
    f = Value(-2.0, label="f")
    L = d * f; L.label = "L"

    # running back prop without performing gradient descent causes gradients to get added to the Values
    # back_propagation(L, perform_gradient_descent=False)
    draw_dot(L).view()
    #
    # back_propagation(L, perform_gradient_descent=True)
    # forward_pass(L)
    #
    # back_propagation(L, perform_gradient_descent=True)
    # forward_pass(L)
    h = 0.001
    gradient_check(L, b, h)

    # draw_dot(L, "Divgraph2.gv").view()


class Operation(Enum):
    PLUS = "+"
    MINUS = "-"
    TIMES = "*"
    DIVIDED_BY = "/"
    EXPONENT = "^"
    TANH = "tanh"
    SUM = "sum"

    @staticmethod
    def operate(operation, values: list):
        if len(values) == 2:
            if operation is Operation.PLUS:
                return values[0] + values[1]
            elif operation is Operation.MINUS:
                return values[0] - values[1]
            elif operation is Operation.TIMES:
                return values[0] * values[1]
            elif operation is Operation.DIVIDED_BY:
                if values[1] == 0:
                    raise ValueError("Cannot divide by zero")
                return values[0] / values[1]
        elif operation is Operation.TANH:
            return numpy.tanh(values[0])
        elif operation is Operation.SUM:
            return sum(values)
        else:
            raise ValueError("Cannot divide by zero")





class Value:
    def __init__(self, data, children=(), operation: Operation = None, label: str = None):
        self.data = data
        self.children = children
        self.operation: Operation = operation

        self._setup_label(label)

        self._calculate_child_gradients = lambda: None
        # The gradient with respect to the loss function (dL/d-self)
        # 'loss' will be the node on which back_propagation() will be called
        # gradients accumulate, so they need to be initialized at 0
        self.gradient_with_respect_to_loss = 0.0

    def _setup_label(self, label):
        if label is not None:
            self.label = label
        elif len(self.children) == 2:
            child1: Value = self.children[0]
            child2 = self.children[1]
            self.label = child1._label(child2, self.operation)
        else:
            self.label = None

    # The string that is print when you do print(object)
    def __repr__(self):
        return f"Value(data={self.data}, label={self.label}, operation={self.operation}, gradient={self.gradient_with_respect_to_loss})"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other, label=f"{other}")
        result = Value(self.data + other.data, (self, other), Operation.PLUS)

        def _gradient_calculation():
            self.gradient_with_respect_to_loss += 1 * result.gradient_with_respect_to_loss
            other.gradient_with_respect_to_loss += 1 * result.gradient_with_respect_to_loss

        result._calculate_child_gradients = _gradient_calculation
        return result

    def __sub__(self, other):
        other = other if isinstance(other, Value) else Value(other, label=f"{other}")
        result = Value(self.data - other.data, (self, other), Operation.MINUS)

        def _gradient_calculation():
            self.gradient_with_respect_to_loss += 1 * result.gradient_with_respect_to_loss
            other.gradient_with_respect_to_loss += -1 * result.gradient_with_respect_to_loss

        result._calculate_child_gradients = _gradient_calculation
        return result


    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other, label=f"{other}")

        # the result is the parent during back prop
        result = Value(self.data * other.data, (self, other), Operation.TIMES)

        def _gradient_calculation():
            # since this is in a block the new_value.gradient_with_respect_to_loss
            # isn't called until it is calculated.
            self.gradient_with_respect_to_loss += other.data * result.gradient_with_respect_to_loss
            other.gradient_with_respect_to_loss += self.data * result.gradient_with_respect_to_loss

        # during backprop the parent (i.e. "new_value") sets the children's gradients
        result._calculate_child_gradients = _gradient_calculation
        return result

    def __truediv__(self, other):
        other = other if isinstance(other, Value) else Value(other, label=f"{other}")
        result = Value(self.data / other.data, (self, other), Operation.DIVIDED_BY, self._label(other, Operation.DIVIDED_BY))

        def _gradient_calculation():
            self.gradient_with_respect_to_loss += 1 / other.data * result.gradient_with_respect_to_loss

            # d / d-other = - a / b^2 (because a/b = a*b^-1)
            other.gradient_with_respect_to_loss -= self.data / (other.data ** 2) * result.gradient_with_respect_to_loss

        result._calculate_child_gradients = _gradient_calculation
        return result

    def _label(self, other, operator: Operation) -> str:
        if (operator == Operation.TIMES or operator == Operation.DIVIDED_BY): # and len(self.label) == 1:
            new_label = f"({self.label}){operator.value}{other.label}"
        else:
            new_label = f"{self.label}{operator.value}{other.label}"
        return new_label

    def __rmul__(self, other):
        other = other if isinstance(other, Value) else Value(other, label=f"{other}")
        # reverse multiply
        # Handles the case: other.__mul__(self)
        # crashes because self.data is needed
        # example: 2 * value
        new_value = self * other
        new_value.label = other._label(self, Operation.TIMES)
        return new_value

    def __radd__(self, other):
        other = other if isinstance(other, Value) else Value(other, label=f"{other}")
        new_value = self + other
        new_value.label = f"{other.label}+{self.label}"
        return new_value

    def __rsub__(self, other):
        other = other if isinstance(other, Value) else Value(other, label=f"{other}")
        new_value = (self - other) * -1
        new_value.label = f"{other.label}-{self.label}"
        return new_value

    def exp(self, other):
        other = other if isinstance(other, Value) else Value(other, label=f"{other}")

        # the result is the parent during back prop
        result = Value(self.data ** other.data, (self, other), Operation.EXPONENT)

        def _gradient_calculation():
            # since this is in a block the new_value.gradient_with_respect_to_loss
            # isn't called until it is calculated.

            if self.data < 0:
                raise ValueError("Negative base not supported for logarithm in derivative calculation.")
                # Or, if using absolute value:
                # log_term = numpy.log(abs(self.data))
            else:
                log_term = numpy.log(self.data)

            # d / d-other = b * a ^ (b-1) (because the derivative of a/b with respect to a is a*b^-1)
            self.gradient_with_respect_to_loss += other.data * self.data ** (other.data - 1)

            # d / d-other = a^b * log(a) (because the derivative of a/b with respect to b is a^b*log(a) )
            other.gradient_with_respect_to_loss += self.data ** other.data * log_term

        # during backprop the parent (i.e. "result") sets the children's gradients
        result._calculate_child_gradients = _gradient_calculation
        return result

    def tanh(self, label=None):
        # y = (e^2x - 1) / (e^2x + 1)
        x = self.data
        y = (numpy.e ** (2 * x) - 1) / (numpy.e ** (2 * x) + 1)

        # the result is the parent during back prop
        result = Value(y, (self, ), Operation.TANH, label)

        def _gradient_calculation():
            # since this is in a block the new_value.gradient_with_respect_to_loss
            # isn't called until it is calculated.

            # dd / dx = 1 - tanh(x)^2
            self.gradient_with_respect_to_loss += 1 - result.data ** 2

        # during backprop the parent (i.e. "result") sets the children's gradients
        result._calculate_child_gradients = _gradient_calculation
        return result

    def _gradient_descent(self, step_size):
        self.data += step_size * self.gradient_with_respect_to_loss


    def back_propagation(self, perform_gradient_descent: bool):
        # Initialize the gradient of the loss function as 1
        self.gradient_with_respect_to_loss = 1

        # Backpropagation: iterate over the graph in reverse (from outputs to inputs)

        # Sort the graph in topological order to ensure proper gradient propagation
        # Essential for correctly applying the chain rule in backpropagation.
        topologically_sorted_graph = topological_sort(self)

        # At each node, apply the chain rule to calculate and accumulate gradients.
        for node in reversed(topologically_sorted_graph):
            node._calculate_child_gradients()

            if perform_gradient_descent and node is not self:
                # gradient descent
                node._gradient_descent(step_size=0.001)


def topological_sort(value: Value) -> ['Value']:
    # sort the graphy such that children get added after parents
    topologically_sorted_graph = []
    visited = set()

    def build_topological_sort(value):
        if value not in visited:
            visited.add(value)
            for child in value.children:
                build_topological_sort(child)
            topologically_sorted_graph.append(value)

    build_topological_sort(value)
    return topologically_sorted_graph


def forward_pass(value, value_to_update: 'Value' = None, h = None):
    topo: [Value] = topological_sort(value)

    # topo is sorted children to parents
    for v in topo:
        if v.operation is not None:
            child1 = v.children[0]
            child2 = v.children[1]
            v.data = Operation.operate(v.operation, child1.data, child2.data)
            if v == value_to_update:
                v.data += h


def gradient_check(loss, value_to_update: 'Value', h):
    L1 = loss.data

    value_to_update.data += h

    forward_pass(loss, value_to_update, h)
    loss.back_propagation(perform_gradient_descent=False)

    L2 = loss.data

    slope = (L2 - L1) / h
    print(slope)


def trace(root):
    # builds a set of all nodes and edges in a graph
    nodes, edges = set(), set()

    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v.children:
                edges.add((child, v))
                build(child)
    build(root)
    return nodes, edges

def draw_dot(root, file_name: str = None):
    dot = Digraph(graph_attr={'rankdir': 'LR'}) # LR = left to right, default format is pdf
    if file_name is not None:
        dot.filename = file_name
    nodes, edges = trace(root)
    for n in nodes:
        uid = str(id(n))
        # for any value in the graph, create a rectangular ('record') node for it
        dot.node(name = uid, label = "{ %s | data %.4f | gradient %.4f }" % (n.label, n.data, n.gradient_with_respect_to_loss), shape='record')
        if n.operation:
            # if this value is a result of some operation, create an op node for it
            dot.node(name = uid + n.operation.value, label = n.operation.value)
            # and connect this node to it
            dot.edge (uid + n.operation.value, uid)
    for n1, n2 in edges:
        # connect n1 to the op node of n2
        dot.edge(str(id(n1)), str(id(n2)) + n2.operation.value)

    return dot


class Neuron:

    def __init__(self, inputs):
        # initialize n weights as random numbers between -1 and 1
        # where n is the number of inputs
        self.weights = [Value(random.uniform(-1,1), label=f"w{i}") for i in range(inputs)]
        self.b = Value(random.uniform(-1,1), label="b")

    def __call__(self, inputs):
        # w * x + b

        zipped_weights_and_input = zip(self.weights, inputs)
        stimulation = self.b.data
        children = [self.b]
        for weight, input_x in zipped_weights_and_input:
            input_x = input_x if isinstance(input_x, Value) else Value(input_x, label=f"{input_x}")
            stimulation_by_single_input = weight * input_x
            stimulation += stimulation_by_single_input.data
            children.append(stimulation_by_single_input)

        activation = Value(stimulation, children=tuple(children), operation=Operation.SUM, label="cell body stimulation")

        def _gradient_calculation():
            for child in children:
                child.gradient_with_respect_to_loss += 1 * activation.gradient_with_respect_to_loss

        activation._calculate_child_gradients = _gradient_calculation
        print(activation)

        out = activation.tanh()
        out.label = "activation"
        print(out)
        return out


class FullyConnectedLayer:

    def __init__(self, number_of_inputs_to_layer, neurons_in_layer):
        self.neurons = [Neuron(number_of_inputs_to_layer) for _ in range(neurons_in_layer)]

    def __call__(self, x_input_to_layer):
        outs = []
        for neuron in self.neurons:
            neuron_output = neuron(x_input_to_layer)
            outs.append(neuron_output)
        return outs[0] if len(outs) == 1 else outs


class MultilayerFullyConnectedNetwork:

    def __init__(self, number_of_inputs, list_of_layer_output_dimensions):

        # layers will live in between the items in the array
        edges_between_layers = [number_of_inputs] + list_of_layer_output_dimensions

        self.layers = []
        # the layers are a list of input -> output pairs for that layer
        for i in range(len(list_of_layer_output_dimensions)):
            layer_input_size = edges_between_layers[i]
            layer_output_size = edges_between_layers[i+1]
            fully_connected_layer = FullyConnectedLayer(layer_input_size, layer_output_size)
            self.layers.append(fully_connected_layer)

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x



if __name__ == '__main__':

    x = [2.0, 3.0, -1.0]

    mlfcn = MultilayerFullyConnectedNetwork(3, [4, 4, 1])
    out = mlfcn(x)
    print(out)
    out.back_propagation(False)
    draw_dot(out).view()

