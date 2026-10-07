#include "nn.hpp"

#include <cstddef>
#include <random>
#include <stdexcept>
#include <vector>

Neuron::Neuron(int nin, std::mt19937& gen, bool nonlin) : b(Value(0.0f)), nonlin(nonlin) {
    if (nin <= 0) {
        throw std::invalid_argument{"nin must be positive"};
    }

    std::uniform_real_distribution<float> dist{-1.0f, 1.0f};

    w.reserve(nin);
    for (int i = 0; i < nin; i++) {
        w.emplace_back(dist(gen));
    }
}

Value Neuron::operator()(const std::vector<Value>& x) const {
    if (x.size() != w.size()) {
        throw std::invalid_argument{"input size must match neuron input count"};
    }

    Value act = b;
    for (std::size_t i = 0; i < x.size(); i++) {
        act = act + w[i] * x[i];
    }
    return nonlin ? act.relu() : act;
}

std::vector<Value> Neuron::parameters() {
    std::vector<Value> params;
    params.reserve(w.size() + 1);

    for (const Value& weight : w) {
        params.push_back(weight);
    }

    params.push_back(b);
    return params;
}

Layer::Layer(int nin, int nout, std::mt19937& gen, bool nonlin) {
    if (nout <= 0) {
        throw std::invalid_argument{"nout must be positive"};
    }

    neurons.reserve(nout);
    for (int i = 0; i < nout; i++) {
        neurons.emplace_back(nin, gen, nonlin);
    }
}

std::vector<Value> Layer::operator()(const std::vector<Value>& x) const {
    std::vector<Value> outs;
    outs.reserve(neurons.size());

    for (const auto& n : neurons) {
        outs.push_back(n(x));
    }

    return outs;
}

std::vector<Value> Layer::parameters() {
    std::vector<Value> params;
    for (Neuron& neuron : neurons) {
        std::vector<Value> neuronParams = neuron.parameters();
        params.insert(params.end(), neuronParams.begin(), neuronParams.end());
    }
    return params;
}

MLP::MLP(int nin, const std::vector<int>& nouts, std::mt19937& gen) {
    if (nouts.empty()) {
        throw std::invalid_argument{"nouts must not be empty"};
    }
    layers.reserve(nouts.size());

    int inputCount = nin;
    for (std::size_t i = 0; i < nouts.size(); i++) {
        layers.emplace_back(inputCount, nouts[i], gen, i != nouts.size() - 1);
        inputCount = nouts[i];
    }
}

std::vector<Value> MLP::operator()(std::vector<Value> x) const {
    for (const auto& layer : layers) {
        x = layer(x);
    }
    return x;
}

std::vector<Value> MLP::parameters() {
    std::vector<Value> params;
    for (Layer& layer : layers) {
        std::vector<Value> layerParams = layer.parameters();
        params.insert(params.end(), layerParams.begin(), layerParams.end());
    }
    return params;
}
