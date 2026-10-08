#ifndef NN_HPP
#define NN_HPP

#include "engine.hpp"

#include <random>
#include <vector>

class Neuron {
  public:
    explicit Neuron(int nin, std::mt19937& gen, bool nonlin = true);
    Value operator()(const std::vector<Value>& x) const;
    std::vector<Value> parameters();

  private:
    std::vector<Value> w;
    Value b;
    bool nonlin;
};

class Layer {
  public:
    explicit Layer(int nin, int nout, std::mt19937& gen, bool nonlin = true);
    std::vector<Value> operator()(const std::vector<Value>& x) const;
    std::vector<Value> parameters();

  private:
    std::vector<Neuron> neurons;
};

class MLP {
  public:
    explicit MLP(int nin, const std::vector<int>& nouts, std::mt19937& gen);
    std::vector<Value> operator()(std::vector<Value> x) const;
    std::vector<Value> parameters();

  private:
    std::vector<Layer> layers;
};

#endif
