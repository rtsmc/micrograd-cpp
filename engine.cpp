#include <cassert>
#include <memory>
#include <optional>
#include <unordered_set>
#include <ostream>
#include <vector>
#include <iostream>
#include <cmath>

enum class Op {
    None,
    Add,
    Mul,
    ReLU,
    Pow,
};

class Value {
    private:
        struct Node {
            float data;
            float grad = 0.0;
            std::optional<float> exponent;

            std::vector<std::shared_ptr<Node>> prev;
            Op op = Op::None;

            Node(float data) : data(data) {};

            void back() {
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
        };

        std::shared_ptr<Node> node;

        explicit Value(std::shared_ptr<Node> n) : node(std::move(n)) {}

        void build_topo(std::vector<Node*>& topo, std::unordered_set<Node*>& visited, Node* v) {
            if (!visited.count(v)) {
                visited.insert(v);
                for (const auto& child : v->prev) {
                    build_topo(topo, visited, child.get());
                }
                topo.push_back(v);
            }
        };


    public:
        Value(float data) : node(std::make_shared<Node>(data)) {}

        void backward() {
            std::vector<Node*> topo = {};
            std::unordered_set<Node*> visited = {};

            build_topo(topo, visited, node.get());

            node->grad = 1.0;
            for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
                (*it)->back();
            }
        }

        void display() {
            std::vector<Node*> topo = {};
            std::unordered_set<Node*> visited = {};

            build_topo(topo, visited, node.get());
            for (int i = 0; i < topo.size(); i++) {
                std::cout << i << ": data: " << topo[i]->data
                    << " grad: " << topo[i]->grad << "\n";
            }
        }

        Value relu() const {
            auto out = std::make_shared<Node>(node->data < 0 ? 0 : node->data);

            out->prev = {node};
            out->op = Op::ReLU;

            return Value(out);
        }

        Value pow(const float other) const {
            auto out = std::make_shared<Node>(std::pow(node->data, other));

            out->prev = {node};
            out->op = Op::Pow;
            out->exponent = other;

            return Value(out);
        };

        Value neg() const { return *this * -1; }

        // operators
        Value operator+(const float other) const { return *this + Value(other); }
        Value operator+(const Value& other) const {
            auto out = std::make_shared<Node>(node->data + other.node->data);

            out->prev = {node, other.node};
            out->op = Op::Add;

            return Value(out);
        }

        Value operator-(float other) const { return *this + -other; };
        Value operator-(const Value& other) const { return *this + other.neg(); };

        Value operator*(const float other) const { return *this * Value(other); }
        Value operator*(const Value& other) const {
            auto out = std::make_shared<Node>(node->data * other.node->data);

            out->prev = {node, other.node};
            out->op = Op::Mul;

            return Value(out);
        }

        Value operator/(const float other) const { return *this * std::pow(other, -1); }
        Value operator/(const Value& other) const { return *this * other.pow(-1); }

        friend std::ostream& operator<<(std::ostream& os, const Value& v) {
            return os << "Value(data=" << v.node->data << ")";
        }
};
