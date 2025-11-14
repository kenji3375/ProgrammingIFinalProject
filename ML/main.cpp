#include <iostream>
#include <vector>
#include <math.h>
#include <random>
#include <iomanip>

inline double relu(double n) {
    return std::max(0.0, n);
}

struct Neuron {
    double weight;
    double bias;

    Neuron(double w=1, double b=0) {
        weight  = w;
        bias    = b;
    }

    double activate(std::vector<double> layer) {
        double sum = 0;
        for(int i=0; i<layer.size(); ++i) {
            sum += weight*layer[i];
        }
        return std::max(0.0, sum);
    }
};

// struct Matrix {
//     std::vector<double> mat;

//     Matrix(int size) {
//         mat = std::vector<double>(size, 0);
//     }
//     Matrix(int size, double values[]) {
//         mat = std::vector<double>(size);
//         int size;
//         for(int i=0; i<sizeof(values)>>3; ++i) {

//         }
//     }
// };

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

        for(int l=0; l<layers; ++l) {
            neurons[l] = std::vector<Neuron>(layerSize[l], Neuron(1,1));
        }
    }

    std::vector<double> out(std::vector<double> input) {
        std::vector<double> prev = input;
        std::vector<double> curr;

        for(int l = 1; l<layers; ++l) {
            curr = std::vector<double>(layerSize[l], 0.0);
            for(int i=0; i<layerSize[l]; ++i) {
                curr[i] = neurons[l][i].activate(prev);
            }
            prev = curr;
        }

        return curr;
    }
};

int main(int argc, char const *argv[])
{
    std::vector<int> layers = {4,4,1};

    MultiLayerPerceptron nn(layers);

    std::vector<double> input = {2,6,2,5};

    std::vector<double> output = nn.out(input);

    for(int i=0; i<output.size(); ++i) {
        std::cout<<std::fixed<<std::setw(5)<<output[i]<<"\n";
    }

    return 0;
}
