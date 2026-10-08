#ifndef ENGINE_HPP
#define ENGINE_HPP

#include <memory>
#include <optional>
#include <ostream>
#include <unordered_set>
#include <vector>

class Value {
  public:
    explicit Value(float data);

    void backward();
    void display() const;

    float data() const;
    float grad() const;

    void setData(float data);
    void zeroGrad();

    Value relu() const;
    Value pow(float other) const;
    Value neg() const;

    Value operator+(float other) const;
    Value operator+(const Value& other) const;

    Value operator-(float other) const;
    Value operator-(const Value& other) const;

    Value operator*(float other) const;
    Value operator*(const Value& other) const;

    Value operator/(float other) const;
    Value operator/(const Value& other) const;

    friend std::ostream& operator<<(std::ostream& os, const Value& v);

  private:
    enum class Op {
        None,
        Add,
        Mul,
        ReLU,
        Pow,
    };

    struct Node {
        float data;
        float grad = 0.0f;
        std::optional<float> exponent;
        std::vector<std::shared_ptr<Node>> prev;
        Op op = Op::None;

        Node(float data);

        void back();
    };

    std::shared_ptr<Node> node;

    explicit Value(std::shared_ptr<Node> n);

    static void build_topo(std::vector<Node*>& topo, std::unordered_set<Node*>& visited, Node* v);
};

#endif
