#include "engine.cpp"
#include <cstdio>
#include <iostream>

void test1() {
    std::cout << "Test1\n";
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
    o.display();
}

void test2() {
    std::cout << "Test2\n";
    Value a(3.0);
    Value b = a + a;
    b.backward();
    b.display();
}

void test3() {
    std::cout << "Test3\n";
    Value a(-2.0);
    Value b(3.0);
    Value d = a * b;
    Value e = a + b;
    Value f = d * e;

    f.backward();
    f.display();
}

void testOverloading() {
    std::cout << "testing ops with Value and a float\n";
    Value a(2.0);
    std::cout << a + 1 << "\n";

    Value b(2.0);
    std::cout << b * 2 << "\n";
}

void testPow() {
    std::cout << "Testing powers\n";
    Value a(4.0);
    Value c = a.pow(2);
    Value o = c.relu();
    std::cout << o << "\n";
    o.backward();
    o.display();
}

void testDiv() {
    std::cout << "Testing div\n";
    Value a(4.0);
    Value b(2.0);
    Value c = a/b;
    std::cout << c << "\n";
    c.backward();
    c.display();
}

void testMinus() {
    std::cout << "Testing subtract\n";
    Value a(6.0);
    Value b(3.0);
    Value c = a - b;
    Value d = c - -1;
    std::cout << d << "\n";
    d.backward();
    d.display();
}

int main() {
    test1();
    std::cout << "------------------------------------------------\n";
    test2();
    std::cout << "------------------------------------------------\n";
    test3();
    std::cout << "------------------------------------------------\n";
    testOverloading();
    std::cout << "------------------------------------------------\n";
    testPow();
    std::cout << "------------------------------------------------\n";
    testDiv();
    std::cout << "------------------------------------------------\n";
    testMinus();
}
