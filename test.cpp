#include "engine.hpp"
#include "nn.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

constexpr float kAbsoluteTolerance{1e-5f};
constexpr float kRelativeTolerance{1e-5f};

void expectNear(const char* label, float actual, float expected) {
    const float difference{std::fabs(actual - expected)};
    const float scale{std::max(std::fabs(actual), std::fabs(expected))};
    const float tolerance{kAbsoluteTolerance + kRelativeTolerance * scale};

    if (std::isfinite(actual) && std::isfinite(expected) && difference <= tolerance) {
        return;
    }

    throw std::runtime_error{std::string{label} + ": expected " + std::to_string(expected) + ", got " + std::to_string(actual)};
}

// testing engine
void testLeafBackward() {
    Value value{3.0f};

    expectNear("leaf data", value.data(), 3.0f);
    expectNear("leaf initial gradient", value.grad(), 0.0f);

    value.backward();

    expectNear("leaf output gradient", value.grad(), 1.0f);
}

void testBinaryArithmetic() {
    {
        Value a{4.0f};
        Value b{2.0f};
        Value output{a + b};

        expectNear("addition value", output.data(), 6.0f);
        output.backward();
        expectNear("addition output gradient", output.grad(), 1.0f);
        expectNear("addition left gradient", a.grad(), 1.0f);
        expectNear("addition right gradient", b.grad(), 1.0f);
    }

    {
        Value a{4.0f};
        Value b{2.0f};
        Value output{a * b};

        expectNear("multiplication value", output.data(), 8.0f);
        output.backward();
        expectNear("multiplication output gradient", output.grad(), 1.0f);
        expectNear("multiplication left gradient", a.grad(), 2.0f);
        expectNear("multiplication right gradient", b.grad(), 4.0f);
    }

    {
        Value a{4.0f};
        Value b{2.0f};
        Value output{a - b};

        expectNear("subtraction value", output.data(), 2.0f);
        output.backward();
        expectNear("subtraction output gradient", output.grad(), 1.0f);
        expectNear("subtraction left gradient", a.grad(), 1.0f);
        expectNear("subtraction right gradient", b.grad(), -1.0f);
    }

    {
        Value a{4.0f};
        Value b{2.0f};
        Value output{a / b};

        expectNear("division value", output.data(), 2.0f);
        output.backward();
        expectNear("division output gradient", output.grad(), 1.0f);
        expectNear("division numerator gradient", a.grad(), 0.5f);
        expectNear("division denominator gradient", b.grad(), -1.0f);
    }
}

void testScalarArithmeticAndNegation() {
    {
        Value value{4.0f};
        Value output{value + 2.0f};
        expectNear("scalar addition value", output.data(), 6.0f);
        output.backward();
        expectNear("scalar addition gradient", value.grad(), 1.0f);
    }

    {
        Value value{4.0f};
        Value output{value * 2.0f};
        expectNear("scalar multiplication value", output.data(), 8.0f);
        output.backward();
        expectNear("scalar multiplication gradient", value.grad(), 2.0f);
    }

    {
        Value value{4.0f};
        Value output{value - 2.0f};
        expectNear("scalar subtraction value", output.data(), 2.0f);
        output.backward();
        expectNear("scalar subtraction gradient", value.grad(), 1.0f);
    }

    {
        Value value{4.0f};
        Value output{value / 2.0f};
        expectNear("scalar division value", output.data(), 2.0f);
        output.backward();
        expectNear("scalar division gradient", value.grad(), 0.5f);
    }

    {
        Value value{4.0f};
        Value output{value.neg()};
        expectNear("negation value", output.data(), -4.0f);
        output.backward();
        expectNear("negation gradient", value.grad(), -1.0f);
    }
}

void testSubtractionChain() {
    Value a{6.0f};
    Value b{3.0f};
    Value intermediate{a - b};
    Value output{intermediate - -1.0f};

    expectNear("subtraction chain intermediate value", intermediate.data(), 3.0f);
    expectNear("subtraction chain value", output.data(), 4.0f);
    output.backward();
    expectNear("subtraction chain output gradient", output.grad(), 1.0f);
    expectNear("subtraction chain left gradient", a.grad(), 1.0f);
    expectNear("subtraction chain right gradient", b.grad(), -1.0f);
}

void testRelu() {
    {
        Value value{-2.0f};
        Value output{value.relu()};
        expectNear("negative ReLU value", output.data(), 0.0f);
        output.backward();
        expectNear("negative ReLU gradient", value.grad(), 0.0f);
    }

    {
        Value value{0.0f};
        Value output{value.relu()};
        expectNear("zero ReLU value", output.data(), 0.0f);
        output.backward();
        expectNear("zero ReLU gradient", value.grad(), 0.0f);
    }

    {
        Value value{2.0f};
        Value output{value.relu()};
        expectNear("positive ReLU value", output.data(), 2.0f);
        output.backward();
        expectNear("positive ReLU gradient", value.grad(), 1.0f);
    }
}

void testPowers() {
    struct PowerCase {
        float base;
        float exponent;
        float expectedValue;
        float expectedGradient;
        const char* label;
    };

    constexpr PowerCase cases[]{
        {4.0f, 2.0f, 16.0f, 8.0f, "square"},      {2.0f, 0.0f, 1.0f, 0.0f, "zeroth power"}, {2.0f, 1.0f, 2.0f, 1.0f, "first power"},      {2.0f, -1.0f, 0.5f, -0.25f, "reciprocal"},
        {4.0f, 0.5f, 2.0f, 0.25f, "square root"}, {0.0f, 2.0f, 0.0f, 0.0f, "zero squared"}, {-2.0f, 3.0f, -8.0f, 12.0f, "negative cube"},
    };

    for (const PowerCase& testCase : cases) {
        Value base{testCase.base};
        Value output{base.pow(testCase.exponent)};

        expectNear(testCase.label, output.data(), testCase.expectedValue);
        output.backward();
        expectNear(testCase.label, base.grad(), testCase.expectedGradient);
    }
}

void testSharedOperands() {
    {
        Value value{3.0f};
        Value output{value + value};
        expectNear("shared addition value", output.data(), 6.0f);
        output.backward();
        expectNear("shared addition gradient", value.grad(), 2.0f);
    }

    {
        Value value{3.0f};
        Value output{value * value};
        expectNear("shared multiplication value", output.data(), 9.0f);
        output.backward();
        expectNear("shared multiplication gradient", value.grad(), 6.0f);
    }
}

void testSharedIntermediate() {
    Value input{3.0f};
    Value intermediate{input * input};
    Value output{intermediate + intermediate};

    expectNear("shared intermediate value", output.data(), 18.0f);
    output.backward();
    expectNear("shared intermediate gradient", intermediate.grad(), 2.0f);
    expectNear("shared intermediate input gradient", input.grad(), 12.0f);
}

void testBranchedGraph() {
    Value a{-2.0f};
    Value b{3.0f};
    Value product{a * b};
    Value sum{a + b};
    Value output{product * sum};

    expectNear("branched graph value", output.data(), -6.0f);
    output.backward();
    expectNear("branched graph output gradient", output.grad(), 1.0f);
    expectNear("branched graph left gradient", a.grad(), -3.0f);
    expectNear("branched graph right gradient", b.grad(), -8.0f);
}

void testNeuronExample() {
    Value x1{2.0f};
    Value x2{0.0f};
    Value w1{-3.0f};
    Value w2{1.0f};
    Value bias{7.0f};
    Value output{(x1 * w1 + x2 * w2 + bias).relu()};

    expectNear("neuron output value", output.data(), 1.0f);
    output.backward();
    expectNear("neuron output gradient", output.grad(), 1.0f);
    expectNear("neuron x1 gradient", x1.grad(), -3.0f);
    expectNear("neuron x2 gradient", x2.grad(), 1.0f);
    expectNear("neuron w1 gradient", w1.grad(), 2.0f);
    expectNear("neuron w2 gradient", w2.grad(), 0.0f);
    expectNear("neuron bias gradient", bias.grad(), 1.0f);
}

void testTemporaryLifetime() {
    Value input{3.0f};
    Value output{(input + 2.0f) * (input - 1.0f)};

    expectNear("temporary graph value", output.data(), 10.0f);
    output.backward();
    expectNear("temporary graph input gradient", input.grad(), 7.0f);
}


// testing nn module
void testNeuron() {
    std::mt19937 gen{42};
    Neuron neuron{2, gen};
    std::vector<Value> params{neuron.parameters()};
    if (params.size() != 3) {
        throw std::runtime_error{"neuron parameter count must be 3"};
    }

    params[0].setData(-3.0f);
    params[1].setData(1.0f);
    params[2].setData(7.0f);
    std::vector<Value> input{Value{2.0f}, Value{0.0f}};
    Value output{neuron(input)};

    expectNear("Neuron output", output.data(), 1.0f);
    output.backward();
    expectNear("Neuron first input gradient", input[0].grad(), -3.0f);
    expectNear("Neuron second input gradient", input[1].grad(), 1.0f);
    expectNear("Neuron first weight gradient", params[0].grad(), 2.0f);
    expectNear("Neuron second weight gradient", params[1].grad(), 0.0f);
    expectNear("Neuron bias gradient", params[2].grad(), 1.0f);

    params[2].setData(5.0f);
    expectNear("Neuron ReLU after bias update", neuron(input).data(), 0.0f);
}

void testLayer() {
    std::mt19937 gen{42};
    Layer layer{2, 2, gen, false};
    std::vector<Value> params{layer.parameters()};
    if (params.size() != 6) {
        throw std::runtime_error{"layer parameter count must be 6"};
    }

    constexpr float data[]{-3.0f, 1.0f, 7.0f, -1.0f, 2.0f, 1.0f};
    for (std::size_t i = 0; i < params.size(); ++i) {
        params[i].setData(data[i]);
    }
    std::vector<Value> output{layer({Value{2.0f}, Value{0.0f}})};
    if (output.size() != 2) {
        throw std::runtime_error{"layer output count must be 2"};
    }

    expectNear("Layer first output", output[0].data(), 1.0f);
    expectNear("Layer linear negative output", output[1].data(), -1.0f);
}

void testMLP() {
    std::mt19937 gen{42};
    MLP model{2, {3, 1}, gen};
    std::vector<Value> params{model.parameters()};
    if (params.size() != 13) {
        throw std::runtime_error{"MLP parameter count must be 13"};
    }

    // Three hidden neurons (two weights and a bias each), then the output neuron.
    constexpr float data[]{1.0f, 0.0f, 0.0f, -1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 2.0f, 3.0f, 4.0f, -10.0f};
    for (std::size_t i = 0; i < params.size(); ++i) {
        params[i].setData(data[i]);
    }
    std::vector<Value> input{Value{2.0f}, Value{0.0f}};
    std::vector<Value> output{model(input)};
    if (output.size() != 1) {
        throw std::runtime_error{"MLP output count must be 1"};
    }

    // Hidden outputs are [2, 0, 1]; the final linear output is -2.
    expectNear("MLP output", output[0].data(), -2.0f);
    output[0].backward();
    expectNear("MLP first input gradient", input[0].grad(), 2.0f);
    expectNear("MLP second input gradient", input[1].grad(), 4.0f);

    constexpr float gradients[]{4.0f, 0.0f, 2.0f, 0.0f, 0.0f, 0.0f, 8.0f, 0.0f, 4.0f, 2.0f, 0.0f, 1.0f, 1.0f};
    for (std::size_t i = 0; i < params.size(); ++i) {
        expectNear("MLP parameter gradient", params[i].grad(), gradients[i]);
    }

    params[12].setData(-9.0f);
    expectNear("MLP output after bias update", model(input)[0].data(), -1.0f);

    MLP singleLayer{1, {2}, gen};
    std::vector<Value> singleParams{singleLayer.parameters()};
    if (singleParams.size() != 4) {
        throw std::runtime_error{"single-layer MLP parameter count must be 4"};
    }
    for (Value& param : singleParams) {
        param.setData(-1.0f);
    }
    std::vector<Value> singleOutput{singleLayer({Value{1.0f}})};
    if (singleOutput.size() != 2) {
        throw std::runtime_error{"single-layer MLP output count must be 2"};
    }
    expectNear("single-layer MLP first linear output", singleOutput[0].data(), -2.0f);
    expectNear("single-layer MLP second linear output", singleOutput[1].data(), -2.0f);
}

void testNNInvalidInputs() {
    struct InvalidShape {
        int nin;
        std::vector<int> nouts;
    };

    const InvalidShape shapes[]{
        {0, {1}}, {-1, {1}}, {1, {}}, {1, {0}}, {1, {-1}}, {1, {2, 0}}, {1, {2, -1}},
    };
    std::mt19937 gen{42};
    for (const InvalidShape& shape : shapes) {
        bool rejected{false};
        try {
            MLP model{shape.nin, shape.nouts, gen};
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        if (!rejected) {
            throw std::runtime_error{"MLP must reject invalid dimensions"};
        }
    }

    MLP model{2, {1}, gen};
    bool rejected{false};
    try {
        model({Value{1.0f}});
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    if (!rejected) {
        throw std::runtime_error{"MLP must reject an incorrect input size"};
    }
}

int main() {
    try {
        testLeafBackward();
        testBinaryArithmetic();
        testScalarArithmeticAndNegation();
        testSubtractionChain();
        testRelu();
        testPowers();
        testSharedOperands();
        testSharedIntermediate();
        testBranchedGraph();
        testNeuronExample();
        testTemporaryLifetime();
        testNeuron();
        testLayer();
        testMLP();
        testNNInvalidInputs();
    } catch (const std::exception& error) {
        std::cerr << "Test failure: " << error.what() << '\n';
        return EXIT_FAILURE;
    }

    std::cout << "All tests passed.\n";
    return EXIT_SUCCESS;
}
