#include "engine.cpp"
#include <cstdio>
#include <iostream>

int main() {
    Value x1(2.0);
    Value x2(0.0);

    Value w1(-3.0);
    Value w2(1.0);

    Value b(6.8813735870195432);

    Value x1w1 = x1*w1;
    Value x2w2 = x2*w2;
    Value x1w1x2w2 = x1w1 + x2w2;
    Value n = x1w1x2w2 + b;
    Value o = n.relu();

    o.backward();

    std::cout << x1 << '\n' << x2 << '\n' << w1 << '\n' << w2 << '\n' << b << '\n' << x1w1 << '\n' << x2w2 << '\n' <<
        x1w1x2w2 << '\n' << n << '\n' << o << '\n';

    std::cout << "GRADS: \n";

    std::cout << x1.grad() << '\n' << x2.grad() << '\n' << w1.grad() << '\n' << w2.grad() << '\n' << b.grad() << '\n' << x1w1.grad()
        << '\n' << x2w2.grad() << '\n' <<
        x1w1x2w2.grad() << '\n' << n.grad() << '\n' << o.grad() << '\n';
}
