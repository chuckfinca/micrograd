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
            return value1 / value2
        # elif self is Operation.PLUS:
        #     return value1 + value2


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
            other.gradient_with_respect_to_loss += self.data * result.gradient_with_respect_to_loss
# 2/3 == 2 * 1/3
        result.calculate_child_gradients = _gradient_calculation
        return result

    def _label(self, other, operator: Operation) -> str:
        if (operator == Operation.TIMES or operator == Operation.DIVIDED_BY): # and len(self.label) == 1:
            new_label = f"({self.label}){operator.value}{other.label}"
        else:
            new_label = f"{self.label}{operator.value}{other.label}"
        return new_label

    def __rmul__(self, other):
        # reverse multiply
        # Handles the case: other.__mul__(self)
        # crashes because self.data is needed
        # example: 2 * value
        new_value = self * other
        new_value.label = other._label(self, Operation.TIMES)
        return new_value

    def __radd__(self, other):
        new_value = self + other
        new_value.label = f"{other.label}+{self.label}"
        return new_value

    def __rsub__(self, other):
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

    visited = set()
    # topo is sorted children to parents
    for v in topo:
        if v.operation is not None and v not in visited:
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
# class Neuron:
#     def __init__(self):


# def local_back_propagation(parent: Value):
#     # we know the parent's local derivative at this point
#     print(f"local_back_propagation for {parent}")
#
#     for child in parent.children:
#         child: Value = child
#
#         if parent.operation == "+":
#             # Add the children together to make the parent
#             # child + child_2 = p
#             # So the derivative of the parent with respect to the first child is
#             # dp / d-child = 1
#
#             # dL/dp = parent.gradient_with_respect_to_loss
#
#             # dL/d-child = dL/dp * dp/d-child
#             child.gradient_with_respect_to_loss += parent.gradient_with_respect_to_loss * 1
#             print(f"        child.gradient = parent.gradient")
#             # if parent.operation == "-":
#             #     # Subtract the children to make the parent
#             #     # child - child_2 = p
#             #     # So the derivative of the parent with respect to the first child is
#             #     # dp / d-child = 1
#             #
#             #     # dL/dp = parent.gradient_with_respect_to_loss
#             #
#             #     # dL/d-child = dL/dp * dp/d-child
#             #     child.gradient_with_respect_to_loss += parent.gradient_with_respect_to_loss * 1
#             #     print(f"        child.gradient = parent.gradient")
#
#         elif parent.operation == "*":
#             # start at 1 since the derivative with respect to the child is 1 for the child term
#             # child = p
#             # dp / dc1 = 1
#             derivative = 1
#
#             # Multiply the children together to determine the over all derivative with respect to a specific child
#             # child * child_2 ... * child_n = p
#             # So the derivative of the parent with respect to the child is
#             # dp / d-child = child_2 ... * child_n
#             for other_child in parent.children:
#                 if other_child != child:
#                     derivative *= other_child.data
#
#             # dL/d-child = dL/dp * dp/d-child
#             child.gradient_with_respect_to_loss += parent.gradient_with_respect_to_loss * derivative
#
#         elif parent.operation == "tanh":
#             # d/dx tanh(x) = 1 - tanh^2(x) = 1 - (tanh(x))^2
#             # d-parent  / d-child = 1 - tanh(child) ** 2
#             # remember that tanh(child) = parent.data
#             child.gradient_with_respect_to_loss += 1 - parent.data ** 2
#         else:
#             raise ValueError("operation not defined")
#
#     for child in parent.children:
#         local_back_propagation(child)


def derivatives():
    h = 0.0001

    # inputs
    a = 2.0
    b = -3.0
    c = 10.0

    # f(x)
    f_of_x = a*b + c

    # derivative with respect to a (df/da)
    # a += h

    # derivative with respect to b (df/db)
    # b += h

    # derivative with respect to c (df/dc)
    c += h

    f_of_x_plus_h = a*b + c

    print(f_of_x)
    print(f_of_x_plus_h)

    slope = (f_of_x_plus_h - f_of_x) / h
    print(slope)

def f(x):
    return 3*x**2 - 4*x + 5

def plot_array():
    # a range from -5 to 5 with increment of 0.25
    xs = numpy.arange(-5, 5, 0.25)
    ys = f(xs)
    print(ys)

    # the amount to nudge x to the right
    h = .001

    xs_nudged_right = xs + h
    ys_nudged_right = f(xs_nudged_right)
    print(ys_nudged_right)

    slope = (f(xs + h) - f(xs)) / h
    print(slope)

    # Setup the plot
    matplotlib_pyplot.plot(xs, ys)

    # Show the plot
    matplotlib_pyplot.show()

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


def gradient_point_check(value: Value, modified_value: Value, h: float):
    print("gradient_point_check----------")
    L = value
    print(L)
    L_children = list(value.children)
    f = L_children[0]
    print(f"f = {f}")
    e = L_children[1]
    print(f"e = {e}")

    e_f = L.operation

    if len(f.children) > 0:
        f_children = list(f.children)
        d = f_children[0]
        c = f_children[1]
        c_d = f.operation
    elif len(e.children) > 0:
        e_children = list(e.children)
        d = e_children[0]
        c = e_children[1]
        c_d = e.operation
    else:
        print("drat")
    print(f"c = {c}")
    print(f"d = {d}")

    if len(d.children) > 0:
        d_children = list(d.children)
        b = d_children[0]
        a = d_children[1]
        a_b = d.operation
    elif len(c.children) > 0:
        c_children = list(c.children)
        b = c_children[0]
        a = c_children[1]
        a_b = c.operation
    else:
        print("dratt")
    print(f"a = {a}")
    print(f"b = {b}")

    print("L:")
    print(L)
    L1 = L.data

    for v in [a, b, c, d, e, f]:
        if modified_value == v:
            print("found ya!")
            v.data += h

    print("L_a:")
    print(a)
    print(b)
    if a_b == "+":
        d = a + b
    else:
        d = a * b
    print(d)

    if c_d == "+":
        e = c + d
    else:
        e = c * d

    if L.operation == "+":
        L = e + f
    else:
        L = e * f

    L2 = L.data

    print(f"a = {a}")
    print(f"b = {b}")
    print(f"c = {c}")
    print(f"d = {d}")
    print(f"e = {e}")
    print(f"f = {f}")
    print(f"==-=-=-==")
    print("L_o:")
    print(value)
    print("L:")
    print(L)
    print(L1)
    print(L2)
    print((L2 - L1) / h)


if __name__ == '__main__':

    # plot_array()
    # derivatives()

    # EXAMPLE for time ~45:00
    # a = Value(2.0, (), "a")
    # b = Value(-3.0, (), "b")
    # c = Value(10.0, (), "c")
    # d = a*b
    # e = d+c
    # f = Value(-2.0, (), "f")

    # Loss function
    # L = e*f
    # L.back_propagation()

    ## a += 0.01 * a.gradient_with_respect_to_loss
    ## b += 0.01 * b.gradient_with_respect_to_loss
    ## c += 0.01 * c.gradient_with_respect_to_loss
    ## f += 0.01 * f.gradient_with_respect_to_loss

    # print(a.data + 0.01 * a.gradient_with_respect_to_loss)
    # print(b.data + 0.01 * b.gradient_with_respect_to_loss)
    # print(c.data + 0.01 * c.gradient_with_respect_to_loss)
    # print(f.data + 0.01 * f.gradient_with_respect_to_loss)
    ## L.gradient = 1
    ## L1 = L.data

    ## h = 0.001
    ## a = Value(2.0 + h, (), "a")
    ## b = Value(-3.0, (), "b")
    ## c = Value(10.0, (), "c")
    ## d = a*b
    ## e = d+c
    ## f = Value(-2.0, (), "f")
    ## L = e*f
    ## L2 = L.data

    ## derivative = (L2 - L1)/h

    ## print(L1)
    ## print(L2)
    ## print(derivative)

    #EXAMPLE for 56:23

    # # inputs x1, x2
    # x1 = Value(2.0, label="x1")
    # x2 = Value(0.0, label="x2")
    #
    # # weights w1, w2
    # w1 = Value(-3.0, label="w1")
    # w2 = Value(1.0, label="w2")
    #
    # # bias of the neuron
    # b = Value(6.8813735870195432, label="b")
    #
    # # x1w1 + x2w2 + b
    # x1w1 = x1 * w1
    # # x1w1.label = "x1w1"
    #
    # x2w2 = x2 * w2
    # # x2w2.label = "x2w2"
    #
    # x1w1x2w2 = x1w1 + x2w2
    # # x1w1x2w2.label = "x1w1 + x2w2"
    #
    # n = x1w1x2w2 + b
    # n.label = "n"
    # o = n.tanh()

    # o.back_propagation()

    # EXAMPLE for 56:23

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