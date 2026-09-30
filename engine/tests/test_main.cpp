#include "test_framework.hpp"
#include <cstdlib>
#include <iostream>

int main() {
    auto& tests = ts_test::get_test_registry();
    // TS_TEST_FILTER=RulesAudit runs only the tests whose name contains the substring.
    const char* filter = std::getenv("TS_TEST_FILTER");
    std::cout << "Running " << tests.size() << " test cases..." << std::endl;

    size_t passed = 0;
    for (const auto& test : tests) {
        if (filter && test.name.find(filter) == std::string::npos) continue;
        std::cout << "[ RUN      ] " << test.name << std::endl;
        test.func();
        std::cout << "[       OK ] " << test.name << std::endl;
        passed++;
    }

    std::cout << "[==========] " << passed << " tests passed." << std::endl;
    return 0;
}
