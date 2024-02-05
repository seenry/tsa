#pragma once

#include "gpu_network.h"

typedef struct SandwichProbeMetadata {
    float mean;
    float variance;
    float welford;
    int samples;
} swpmd_t;

// Topology Extractor
class TExtor {
public:
    void Initialize(GPUNetwork* network);
    void Destroy();

    void Extract(int points_per_pair) {
        ProbeNetwork(points_per_pair);
        Characterize();
        Search();
    };
private:
    void ProbeNetwork(int ppp);
    void Characterize();
    void Search();

    void SingleProbe(int src, int dst_a, int dst_b);

    GPUNetwork* net_;
    swpmd_t* sandwich_data_;
};
