#include <memory>
#include <unordered_set>
#include <ostream>
#include <vector>

enum class Op {
    None,
    Add,
    Mul,
    ReLU
};

class Value {
    private:
        struct Node {
            float data;
            float grad = 0.0;

            std::vector<std::shared_ptr<Node>> prev;
            Op op = Op::None;

            Node(float data) : data(data) {};

            void back() {
                switch (op) {
                    case Op::Add:
                        prev[0]->grad += grad;
                        prev[1]->grad += grad;
                        break;

                    case Op::Mul:
                        prev[0]->grad += prev[1]->data * grad;
                        prev[1]->grad += prev[0]->data * grad;
                        break;

                    case Op::ReLU:
                        prev[0]->grad += (data > 0.0f) * grad;
                        break;

                    default:
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

        float data() const { return node->data; }

        float grad() const { return node->grad; }

        Value relu() const {
            auto out = std::make_shared<Node>(node->data < 0 ? 0 : node->data);

            out->prev = {node};
            out->op = Op::ReLU;

            return Value(out);
        }

        // operators
        Value operator+(const Value& other) const {
            auto out = std::make_shared<Node>(node->data + other.node->data);

            out->prev = {node, other.node};
            out->op = Op::Add;

            return Value(out);
        }

        Value operator*(const Value& other) const {
            auto out = std::make_shared<Node>(node->data * other.node->data);

            out->prev = {node, other.node};
            out->op = Op::Mul;

            return Value(out);
        }

        friend std::ostream& operator<<(std::ostream& os, const Value& v) {
            return os << "Value(data=" << v.node->data << ")";
        }
};
