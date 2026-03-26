#include <oneapi/dpl/execution>
#include <oneapi/dpl/algorithm>

int main(){
    std::vector<int> vec(1000);
    dpl::fill(dpl::execution::dpcpp_default,
    vec.begin(), vec.end(), 42);
    return 0;
}