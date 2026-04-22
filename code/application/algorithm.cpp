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

        long long chunk = remaining; // start greedy: give it everything
        for (int a = 0; a < p; a++)
        {
            if (a == counter) continue;
            chunk = std::min(chunk, std::max(1LL, remaining / p));
        }

        processors[counter].workLoad += chunk;
        remaining -= chunk;
    }

    for (int a = 0; a < p; a++)
        ni[a] = processors[a].workLoad;
}

std::vector<long long> functional_partition(long long n){
    processor GPU_thrust1, GPU_thrust2, CPU_tbb;
    std::vector<long long> returnValues(3);

    GPU_thrust1.name = "GPU";
    GPU_thrust1.sizes = {32000, 64000, 128000, 256000, 512000, 1024000, 2048000,
                         4096000, 8192000, 16384000, 32768000, 65536000,
                         131072000, 262144000, 524288000};
    GPU_thrust1.speeds = {384, 723, 1267, 1582, 2488, 3335, 3764, 4145, 4437,
                          3799, 3419, 3200, 2824, 3025, 3182};

    GPU_thrust2.name = "GPU";
    GPU_thrust2.sizes = GPU_thrust1.sizes;
    GPU_thrust2.speeds = GPU_thrust1.speeds;

    CPU_tbb.name = "CPU";
    CPU_tbb.sizes = {32000, 64000, 128000, 256000, 512000, 1024000, 2048000,
                     4096000, 8192000, 16384000, 32768000, 65536000,
                     131072000, 262144000, 524288000};
    CPU_tbb.speeds = {558, 912, 1009, 1233, 1203, 1301, 1324, 1303, 1271,
                      1039, 793, 681, 651, 637, 652};

    std::vector<processor> processors = {GPU_thrust1, GPU_thrust2, CPU_tbb};
    std::vector<long long> ni = algorithm1_1(n, processors);

    for (int i = 0; i < (int)ni.size(); i++)
        processors[i].workLoad = ni[i];

    algorithm1_2(n, ni, processors);

    long long totalSum = 0;
    for (int a = 0; a < (int)processors.size(); a++){
        totalSum += processors[a].workLoad;
    }

    //if (totalSum == n){
    //    printf("Fully distributed\n");
    //}else{
    //    printf("not fully distributed (sum=%lld, expected=%lld)\n\n", totalSum, n);
    //}

    //printf("final distribution: %lld\n", n);
    //for (int a = 0; a < (int)ni.size(); a++){
    //    printf("%s: %lld\n", processors[a].name.c_str(), processors[a].workLoad);
    //    returnValues.at(a)=processors[a].workLoad;
    //}
    //printf("\n\n");

    return returnValues;
}