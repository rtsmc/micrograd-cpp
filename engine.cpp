#include "engine.hpp"

#include <cassert>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <memory>
#include <ostream>
#include <unordered_set>
#include <utility>
#include <vector>

Value::Value(std::shared_ptr<Node> n) : node(std::move(n)) {}
Value::Value(float data) : node(std::make_shared<Node>(data)) {}

void Value::backward() {
    std::vector<Node*> topo = {};
    std::unordered_set<Node*> visited = {};

    build_topo(topo, visited, node.get());

    node->grad = 1.0f;
    for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
        (*it)->back();
    }
}

void Value::display() const {
    std::vector<Node*> topo;
    std::unordered_set<Node*> visited;

    build_topo(topo, visited, node.get());
    std::size_t i = 0;
    for (Node* nodeptr : topo) {
        std::cout << i++ << ": data: " << nodeptr->data << " grad: " << nodeptr->grad << "\n";
    }
}

float Value::data() const { return node->data; }
float Value::grad() const { return node->grad; }

Value Value::relu() const {
    auto out = std::make_shared<Node>(node->data < 0 ? 0 : node->data);

    out->prev = {node};
    out->op = Op::ReLU;

    return Value(out);
}

Value Value::pow(const float other) const {
    auto out = std::make_shared<Node>(std::pow(node->data, other));

    out->prev = {node};
    out->op = Op::Pow;
    out->exponent = other;

    return Value(out);
}

Value Value::neg() const { return *this * -1; }

Value Value::operator+(const float other) const { return *this + Value(other); }
Value Value::operator+(const Value& other) const {
    auto out = std::make_shared<Node>(node->data + other.node->data);

    out->prev = {node, other.node};
    out->op = Op::Add;

    return Value(out);
}

Value Value::operator-(float other) const { return *this + -other; }
Value Value::operator-(const Value& other) const { return *this + other.neg(); }

Value Value::operator*(const float other) const { return *this * Value(other); }
Value Value::operator*(const Value& other) const {
    auto out = std::make_shared<Node>(node->data * other.node->data);

    out->prev = {node, other.node};
    out->op = Op::Mul;

    return Value(out);
}

Value Value::operator/(const float other) const { return *this * std::pow(other, -1); }
Value Value::operator/(const Value& other) const { return *this * other.pow(-1); }

std::ostream& operator<<(std::ostream& os, const Value& v) { return os << "Value(data=" << v.node->data << ")"; }

void Value::build_topo(std::vector<Node*>& topo, std::unordered_set<Node*>& visited, Node* v) {
    if (!visited.count(v)) {
        visited.insert(v);
        for (const auto& child : v->prev) {
            build_topo(topo, visited, child.get());
        }
        topo.push_back(v);
    }
}

// Node
Value::Node::Node(float data) : data(data) {}

void Value::Node::back() {
    switch (op) {
        case Op::None:
            break;

        case Op::Add:
            assert(prev.size() == 2);
            prev[0]->grad += grad;
            prev[1]->grad += grad;
            break;

        case Op::Mul:
            assert(prev.size() == 2);
            prev[0]->grad += prev[1]->data * grad;
            prev[1]->grad += prev[0]->data * grad;
            break;

        case Op::ReLU:
            assert(prev.size() == 1);
            prev[0]->grad += (data > 0.0f) * grad;
            break;

        case Op::Pow:
            assert(prev.size() == 1);
            assert(exponent.has_value());
            float p = *exponent;
            prev[0]->grad += (p * std::pow(prev[0]->data, p - 1)) * grad;
            break;
    }
}
