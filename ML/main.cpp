#include <iostream>
#include <vector>
#include <math.h>
#include <random>
#include <iomanip>

#define E 2.718281828459


std::vector<double> softmaxLayer(std::vector<double> output) {
    std::vector<double> probabilities(output.size());

    double denominator{};

    for (int i=0; i<output.size(); ++i) {
        denominator += std::pow(E, output[i]);
    }

    for (int i=0; i<output.size(); ++i) {
        probabilities[i] = std::pow(E, output[i])/denominator;
    }

    return probabilities;
}

inline double relu(double n) {
    return std::max(0.0, n);
}

inline double noActivation(double x) {
    return x;
}

inline double fastSigmoid(double x) {
    // return (x/(1+abs(x)))/2 + 0.5;      // not exactly sigmod but is functionally the same
    return (1/(1+std::pow(E, -x)));      // exactly sigmod 
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

    double (*activation)(double);


    
    Neuron(int weightsNumber, double (*act)(double)=&relu) {
        weights = std::vector<double>(weightsNumber, randomDouble(-2,2,2));
        bias    = randomDouble(-2,2,2);

        activation = act;
    }

    double activate(std::vector<double> layer) {
        double sum = 0;
        for(int i=0; i<layer.size(); ++i) {
            sum += layer[i] * weights[i];
        }
        sum += bias;
        return activation(sum);
    }
    
    double lastSum;
    double lastActivation;

    double activateTraining(std::vector<double> layer) {
        double sum = 0;
        for(int i=0; i<layer.size(); ++i) {
            sum += layer[i] * weights[i];
        }
        sum += bias;

        lastSum = sum;
        lastActivation = activation(sum);
        return lastActivation;
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

        for(int l=1; l<layers-1; ++l) {
            neurons[l] = std::vector<Neuron>(layerSize[l], Neuron(layerSize[l-1]));
            for(int n=0; n<layerSize[l]; ++n) {
                neurons[l][n] = Neuron(layerSize[l-1]);
            }
        }
        neurons[layers-1] = std::vector<Neuron>(layerSize[layers-1], Neuron(layerSize[layers-2]));
        for(int n=0; n<layerSize[layers-1]; ++n) {
            neurons[layers-1][n] = Neuron(layerSize[layers-2], noActivation);
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
            return softmaxLayer(tempLayer);

        } else {
            return std::vector<double>(1,-1);
        }
    }

    std::vector<double> forwardPass(std::vector<double> input) {    //same as out, but saves results for each neuron

        if(input.size() == layerSize[0]) {

            std::vector<double> tempLayer;

            for(int L=1; L<layers; ++L) {
                tempLayer = std::vector<double>(layerSize[L]);
                for(int n=0; n<layerSize[L]; ++n) {
                    tempLayer[n] = neurons[L][n].activateTraining(input);
                }
                input = tempLayer;
            }
            return softmaxLayer(tempLayer);

        } else {
            return std::vector<double>(1,-1);
        }
    }

    double loss(std::vector<double> input, std::vector<double> expected) {  //  MSE - mean squared error
        double loss{};
        std::vector<double> res;
        res = this->out(input);

        for(int i=0; i<res.size(); ++i) {
            loss += (res[i]*res[i])/2;
        }

        return loss;
    }

    void backpropagate(std::vector<double> input, std::vector<double> expected, double learningRate, bool showdata=false) {
        std::vector<double> output = out(input);
        double loss = this->loss(output, expected);

        std::vector<std::vector<double>> deltas(layers);

        for(int i=0; i<layers; ++i) {
            deltas[i] = std::vector<double>(layerSize[i], 0);
        }

        for(int i=0; i<deltas.size(); ++i) {    // because softmax is used instead of ReLU, this layer will be treated differently
            deltas[layers-1][i] = output[i] - expected[i];
        }
        
        // deltas for hidden layers
        double sum;
        for(int L=layers-2; L>0; --L) {
            for(int i=0; i<layerSize[L]; ++i) {
                sum=0;
                for(int k=0; k<layerSize[L+1]; ++k) {
                    sum += neurons[L+1][k].weights[i] * deltas[L+1][k];
                }
                // sum *= neurons[L][i].lastSum's derivative //it's always 1
                deltas[L][i] = sum;
            }
        }

        //gradient will not be calculated separately
        // std::vector<std::vector<std::vector<double>>> weight_gradient = std::vector<std::vector<std::vector<double>>>(layers);
        //std::vector<std::vector<double>> bias_gradient   = ;

        for(int n=0; n<layerSize[1]; ++n) {
            for(int w=0; w<layerSize[0]; ++w) {
                neurons[1][n].weights[w] -= learningRate * deltas[1][n] * input[w];
            }
            neurons[1][n].bias -= learningRate * deltas[1][n];
        }
        for(int L=2; L<layers; ++L) {
            for(int n=0; n<layerSize[L]; ++n) {
                for(int w=0; w<layerSize[L-1]; ++w) {
                    neurons[L][n].weights[w] -= learningRate * deltas[L][n] * neurons[L-1][n].lastActivation;
                }
                neurons[L][n].bias -= learningRate * deltas[L][n];
            }
        }
    }

    void trainBatch(std::vector<std::vector<std::vector<double>>> batch, double learningRate) {
        for(int epoch=0; epoch<batch.size(); ++epoch) {
            backpropagate(batch[epoch][0], batch[epoch][1], learningRate);
        }
        std::cout<<"Training on batch finished. \n";
    }
};


// void test() {
//     std::vector<int> layers = {4,4,2};

//     MultiLayerPerceptron nn(layers);

//     std::vector<double> input = {2,6,2,5};

//     std::vector<double> output = nn.out(input);

//     for(int i=0; i<output.size(); ++i) {
//         std::cout<<std::fixed<<std::setw(5)<<output[i]<<"\n";
//     }
// }
void test() {
    std::vector<int> layers = {4,7,5};

    MultiLayerPerceptron nn(layers);

    std::vector<double> input = {2,6,2,5};
    
    std::vector<double> output = nn.out(input);
    
    for(int i=0; i<output.size(); ++i) {
        std::cout<<std::fixed<<std::setw(5)<<output[i]<<"\n";
    }
    std::vector<double> expected = {1, 0, 0, 0, 0, 0, 0};

    double loss = nn.loss(input, expected);
    std::cout<<"loss: "<<loss<<"\n";
}

int main(int argc, char const *argv[])
{
    test();
    // std::cout<<"\n\n";
    // test();
    // std::cout<<"\n\n";
    // test();
    // std::cout<<"\n\n";
    return 0;
}
