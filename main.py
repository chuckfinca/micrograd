import math
from enum import Enum

import matplotlib.pyplot as matplotlib_pyplot
import numpy
from graphviz import Digraph


class Operation(Enum):
    PLUS = "+"
    MINUS = "-"
    TIMES = "*"
    DIVIDED_BY = "/"

    @staticmethod
    def operate(operation, value1, value2):
        if operation is Operation.PLUS:
            return value1 + value2
        elif operation is Operation.MINUS:
            return value1 - value2
        elif operation is Operation.TIMES:
            return value1 * value2
        elif operation is Operation.DIVIDED_BY:
            if value2 == 0:
                raise ValueError("Cannot divide by zero")
            return value1 / value2


class Value:
    def __init__(self, data, children=(), operation: Operation = None, label: str = None):
        self.data = data
        self.children = children
        self.operation: Operation = operation

        self._setup_label(label)

        self.calculate_child_gradients = lambda: None
        # The gradient with respect to the loss function (dL/d-self)
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

        result.calculate_child_gradients = _gradient_calculation
        return result

    def __sub__(self, other):
        other = other if isinstance(other, Value) else Value(other, label=f"{other}")
        result = Value(self.data - other.data, (self, other), Operation.MINUS)

        def _gradient_calculation():
            self.gradient_with_respect_to_loss += 1 * result.gradient_with_respect_to_loss
            other.gradient_with_respect_to_loss += -1 * result.gradient_with_respect_to_loss

        result.calculate_child_gradients = _gradient_calculation
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
        result.calculate_child_gradients = _gradient_calculation
        return result

    def __truediv__(self, other):
        other = other if isinstance(other, Value) else Value(other, label=f"{other}")
        result = Value(self.data / other.data, (self, other), Operation.DIVIDED_BY, self._label(other, Operation.DIVIDED_BY))

        def _gradient_calculation():
            self.gradient_with_respect_to_loss += 1 / other.data * result.gradient_with_respect_to_loss

            # d / d-other = - a / b^2 (because a/b = a*b^-1)
            other.gradient_with_respect_to_loss -= self.data / (other.data ** 2) * result.gradient_with_respect_to_loss

        result.calculate_child_gradients = _gradient_calculation
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

    def tanh(self):
        x = self.data
        t = (math.exp(2*x) - 1) / (math.exp(2*x) + 1)
        # (self, ) means that self is the only child of the new tanh Value object
        return Value(t, (self, ), "tanh")

    def gradient_descent(self, step_size):
        self.data += step_size * self.gradient_with_respect_to_loss


def back_propagation(loss: Value, perform_gradient_descent: bool):
    # Initialize the gradient of the loss function as 1
    loss.gradient_with_respect_to_loss = 1

    # Backpropagation: iterate over the graph in reverse (from outputs to inputs)

    # Sort the graph in topological order to ensure proper gradient propagation
    # Essential for correctly applying the chain rule in backpropagation.
    topologically_sorted_graph = topological_sort(loss)

    # At each node, apply the chain rule to calculate and accumulate gradients.
    for node in reversed(topologically_sorted_graph):
        node.calculate_child_gradients()

        if perform_gradient_descent and node is not loss:
            # gradient descent
            node.gradient_descent(step_size=0.001)


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
    back_propagation(loss, perform_gradient_descent=False)

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


if __name__ == '__main__':

    # inputs x1, x2
    a = Value(2.0, label="a")
    b = Value(-3.0, label="b")
    c = Value(10.0, label="c")
    e = a * b; e.label = "e"
    d = e + c; d.label = "d"
    f = Value(-2.0, label="f")
    L = d * f; L.label = "L"

    back_propagation(L, perform_gradient_descent=False)
    draw_dot(L).view()

    back_propagation(L, perform_gradient_descent=True)
    forward_pass(L)

    back_propagation(L, perform_gradient_descent=True)
    forward_pass(L)
    # h = 0.001
    # gradient_check(L, d, h)

    draw_dot(L, "Divgraph2.gv").view()
    # gradient_point_check(L, a, 0.0001)