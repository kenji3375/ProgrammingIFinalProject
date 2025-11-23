#include <iostream>
#include <vector>
#include <math.h>
#include <random>
#include <iomanip>

inline double relu(double n) {
    return std::max(0.0, n);
}

inline double randomDouble(double min, double max, int p) {
    static thread_local std::mt19937_64 rng(std::random_device{}());
    
    // Scale factor for the desired precision
    double scale = std::pow(10.0, p);

    // Integer distribution for scaled range
    std::uniform_int_distribution<long long> dist(
        static_cast<long long>(std::round(min * scale)),
        static_cast<long long>(std::round(max * scale))
    );

    // Convert back to double with p-decimal precision
    return dist(rng) / scale;
}

struct Neuron {
    double bias;
    std::vector<double> weights;
    
    Neuron(int weightsNumber) {
        weights = std::vector<double>(weightsNumber, randomDouble(1,2,2));
        bias    = randomDouble(0.5,2,2);
    }

    double activate(std::vector<double> layer) {
        double sum = 0;
        for(int i=0; i<layer.size(); ++i) {
            sum += layer[i] * weights[i];
        }
        sum += bias;
        return relu(sum);
    }
};



struct MultiLayerPerceptron {
    std::vector<std::vector<Neuron>> neurons;
    int layers;
    std::vector<int> layerSize;
    
    MultiLayerPerceptron(std::vector<int> matrix) {
        this->layers = matrix.size();
        layerSize = std::vector<int>(layers);
        for(int i=0; i<layers; ++i) {
            layerSize[i] = matrix[i];
            // layerSize.emplace_back(matrix[i]);
        }

        neurons = std::vector<std::vector<Neuron>>(layers);

        // neurons[0] = std::vector<Neuron>(layerSize[0], Neuron(1,1,1)); //first layer will be never used anyway

        for(int l=1; l<layers; ++l) {
            neurons[l] = std::vector<Neuron>(layerSize[l], Neuron(layerSize[l-1]));
            for(int n=0; n<layerSize[l]; ++n) {
                neurons[l][n] = Neuron(layerSize[l-1]);
            }
        }
    }

    std::vector<double> out(std::vector<double> input) {

        if(input.size() == layerSize[0]) {

            std::vector<double> tempLayer;

            for(int L=1; L<layers; ++L) {
                tempLayer = std::vector<double>(layerSize[L]);
                for(int n=0; n<layerSize[L]; ++n) {
                    tempLayer[n] = neurons[L][n].activate(input);
                }
                input = tempLayer;
            }
            return tempLayer;

        } else {
            return std::vector<double>(1,0);
        }
    }
};


void test() {
    std::vector<int> layers = {4,4,2};

    MultiLayerPerceptron nn(layers);

    std::vector<double> input = {2,6,2,5};

    std::vector<double> output = nn.out(input);

    for(int i=0; i<output.size(); ++i) {
        std::cout<<std::fixed<<std::setw(5)<<output[i]<<"\n";
    }
}

int main(int argc, char const *argv[])
{
    test();
    std::cout<<"\n\n";
    test();
    std::cout<<"\n\n";
    test();
    std::cout<<"\n\n";
    return 0;
}
