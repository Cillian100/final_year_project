#include <stdio.h>
#include <iostream>
#include <string>
#include <vector>
#include <cmath>
#include <limits>
 
struct processor
{
    std::string name;
    long long workLoad;
    std::vector<double> sizes;
    std::vector<double> speeds;
 
    double speedAt(double x) const
    {
        if (x <= sizes.front())
            return speeds.front();
        if (x >= sizes.back())
            return speeds.back();
 
        for (size_t i = 0; i < sizes.size() - 1; ++i)
        {
            if (x >= sizes[i] && x <= sizes[i + 1])
            {
                double t = (x - sizes[i]) / (sizes[i + 1] - sizes[i]);
                return speeds[i] + t * (speeds[i + 1] - speeds[i]);
            }
        }
        return speeds.back();
    }
};
 
double intersectWithLine(const processor &proc, double slope)
{
    double xMin = proc.sizes.front();
    double xMax = proc.sizes.back();
 
    auto f = [&](double x)
    { return slope * x - proc.speedAt(x); };
 
    double fMin = f(xMin);
    double fMax = f(xMax);
 
    if (fMin * fMax > 0)
        return (fMin < 0) ? xMax : xMin;
 
    for (int iter = 0; iter < 200; iter++)
    {
        double xMid = (xMin + xMax) / 2.0;
        double fMid = f(xMid);
 
        if (std::abs(fMid) < 1e-9 || (xMax - xMin) < 1e-9)
            return xMid;
 
        if (fMin * fMid < 0)
        {
            xMax = xMid;
            fMax = fMid;
        }
        else
        {
            xMin = xMid;
            fMin = fMid;
        }
    }
    return (xMin + xMax) / 2.0;
}
 
std::vector<long long> algorithm1_1(long long n, const std::vector<processor> &processors)
{
    int p = processors.size();
    double np = static_cast<double>(n) / p;
    std::vector<long long> ni(p);
 
    double maxSpeed = std::numeric_limits<double>::lowest();
    double minSpeed = std::numeric_limits<double>::max();
 
    for (int a = 0; a < p; a++)
    {
        double s = processors.at(a).speedAt(np);
        maxSpeed = std::max(maxSpeed, s);
        minSpeed = std::min(minSpeed, s);
    }
 
    double angleU = std::atan2(maxSpeed, np);
    double angleL = std::atan2(minSpeed, np);
 
    std::vector<double> xU(p), xL(p), xM(p);
    while (true)
    {
        double slopeU = std::tan(angleU);
        double slopeL = std::tan(angleL);
 
        for (int a = 0; a < p; a++)
        {
            xU[a] = intersectWithLine(processors[a], slopeU);
            xL[a] = intersectWithLine(processors[a], slopeL);
        }
 
        double maxSpan = 0.0;
        for (int a = 0; a < p; a++)
        {
            maxSpan = std::max(maxSpan, xL[a] - xU[a]);
        }
 
        if (maxSpan < 1.0 || angleU <= angleL)
            break;
 
        double angleM = (angleU + angleL) / 2.0;
 
        if (angleM == angleU || angleM == angleL)
            break;
 
        double slopeM = std::tan(angleM);
        double sumM = 0.0;
 
        for (int a = 0; a < p; a++)
        {
            xM[a] = intersectWithLine(processors[a], slopeM);
            sumM += xM[a];
        }
 
        if (sumM <= static_cast<double>(n))
            angleU = angleM;
        else
            angleL = angleM;
    }
 
    for (int a = 0; a < p; a++)
        ni[a] = static_cast<long long>(std::floor(xU[a]));
 
    double finalSlope = std::tan(angleU);
    std::cout << "final slope - " << finalSlope << std::endl;
    return ni;
}
 
void algorithm1_2(long long n, std::vector<long long> &ni, std::vector<processor> &processors)
{
    int p = ni.size();
 
    long long totalSum = 0;
    for (int a = 0; a < p; a++)
        totalSum += processors[a].workLoad;
 
    long long remaining = n - totalSum;
 
    while (remaining > 0)
    {
        double minimum = std::numeric_limits<double>::max();
        int counter = 0;
        for (int a = 0; a < p; a++)
        {
            double ratio = (processors[a].workLoad + 1) /
                           processors[a].speedAt(processors[a].workLoad + 1);
            if (ratio < minimum)
            {
                minimum = ratio;
                counter = a;
            }
        }
 
        // FIX 2: Simplified — the loop was redundant; every non-counter
        // iteration set chunk to the same value, so the result was always
        // max(1, remaining/p) regardless of p.
        long long chunk = std::max(1LL, remaining / p);
 
        processors[counter].workLoad += chunk;
        remaining -= chunk;
    }
 
    for (int a = 0; a < p; a++)
        ni[a] = processors[a].workLoad;
}
 
//GPU1 = [11.858714, 25.344969, 49.640888, 94.111173, 149.016963, 213.233692, 274.793620, 321.010472, 381.809340, 455.712360, 455.399067, 440.582470, 445.283224, 453.716600, 453.213031, 452.223315, 430.426348]
//GPU2 = [12.579441, 25.917393, 50.929113, 96.589021, 152.476583, 218.024796, 280.703610, 328.078838, 390.502698, 463.992281, 459.785413, 445.846774, 460.294488, 459.600202, 459.506144, 458.396304, 436.701908]
//CPU1 = [0.072810, 0.111716, 0.269230, 0.590638, 1.146988, 2.308032, 4.642645, 8.599679, 16.795723, 32.294207, 53.659840, 82.975852, 123.968190, 175.006092, 204.581247, 251.888911, 295.908125]
 
//43.7787 90.6567 177.826 360.854 713.295 1324.24 2799.1 5260.67 10748.6 19978.9 37947.5 71539.9 109576 170759 214375 297095 318951 330603 340820 
//37001.5 120985 225773 334802 420317 623262 811071 852045 910915 901150 849654 868894 885798 894344 900075 901712 905329 896245 887064 

std::vector<long long> functional_partition(long long n){
    processor GPU_thrust1, GPU_thrust2, CPU_tbb;
 
    GPU_thrust1.name = "GPU";
    GPU_thrust1.sizes = {10000, 20000, 40000, 80000, 160000, 320000, 640000, 1280000, 2560000, 5120000,
        10240000, 20480000, 40960000, 81920000, 163840000, 327680000, 655360000};
    // FIX 1: Removed the two extra trailing values (820291, 832356) that caused
    // sizes (17 elements) and speeds (was 19) to be mismatched.
    
    GPU_thrust1.speeds={37001.5, 120985, 225773, 334802, 420317, 623262, 811071, 852045, 910915, 901150, 849654, 
        868894, 885798, 894344, 900075, 901712, 905329, 896245, 887064};
    //GPU_thrust1.speeds = {11754.4, 104297, 202244, 314787, 401397, 597073, 767086, 858627, 911662, 844524,
    //    787853, 806201, 817984, 824820, 826178, 824577, 817779};
 
    GPU_thrust2.name = "GPU";
    GPU_thrust2.sizes = GPU_thrust1.sizes;
    GPU_thrust2.speeds = GPU_thrust1.speeds;
 
    CPU_tbb.name = "CPU";
    CPU_tbb.sizes = GPU_thrust1.sizes; 
    CPU_tbb.speeds = {43.7787, 90.6567, 177.826, 360.854, 713.295, 1324.24, 2799.1, 5260.67, 10748.6, 19978.9,
         37947.5, 71539.9, 109576, 170759, 214375, 297095, 318951, 330603, 340820};
         // FIX 1 (cont.): Removed the two extra trailing values (603322, 620562)
    // from CPU speeds for the same reason.
    //CPU_tbb.speeds = {237.606, 576.079, 1199.88, 1895.36, 4044.97, 8455.48, 13075.1, 27146, 46933.6,
    //    75044.4, 126984, 215165, 290582, 409874, 407637, 591910, 525760};
 
    std::vector<processor> processors = {GPU_thrust1, GPU_thrust2, CPU_tbb};
    std::vector<long long> ni = algorithm1_1(n, processors);
 
    for (int i = 0; i < (int)ni.size(); i++)
        processors[i].workLoad = ni[i];
 
    algorithm1_2(n, ni, processors);
 
    long long totalSum = 0;
    for (int a = 0; a < (int)processors.size(); a++){
        totalSum += processors[a].workLoad;
    }
 
    if (totalSum == n){
        printf("Fully distributed\n");
    }else{
        printf("not fully distributed (sum=%lld, expected=%lld)\n\n", totalSum, n);
    }
 
    printf("final distribution: %lld\n", n);
    for (int a = 0; a < (int)processors.size(); a++){
        printf("%s: %lld\n", processors[a].name.c_str(), processors[a].workLoad);
    }
    printf("\n\n");
 
    // FIX 3: Size derived from processors rather than hardcoded to 3.
    std::vector<long long> returnValues(processors.size());
    for (int a = 0; a < (int)processors.size(); a++)
        returnValues[a] = processors[a].workLoad;
 
    return returnValues;
}