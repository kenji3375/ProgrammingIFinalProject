#include <iostream>
#include <vector>
#include <math.h>
#include <random>
#include <iomanip>

#include <fstream>
#include <cstdint>
#include <algorithm>


#define E 2.718281828459


std::vector<double> softmaxLayer(std::vector<double> output) {
    std::vector<double> probabilities(output.size());

    double maxv = *std::max_element(output.begin(), output.end());

    // for(int i=0; i<output.size(); ++i) {
    //     output[i] -= maxv;
    // }

    for(double &v : output) v -= maxv;

    double denominator{};

    for (int i=0; i<output.size(); ++i) {
        denominator += std::exp(output[i]);
    }

    for (int i=0; i<output.size(); ++i) {
        probabilities[i] = std::exp(output[i])/denominator;
    }

    return probabilities;
}

inline double relu(double n) {
    return std::max(0.0, n);
}

inline double noActivation(double x) {
    return x;
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
        double limit = std::sqrt(2.0/weightsNumber);
        weights = std::vector<double>(weightsNumber, randomDouble(-limit,limit,4));
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
            // return tempLayer;
            return softmaxLayer(tempLayer);

        } else {
            return std::vector<double>(1,-1);
        }
    }

    

    double loss(std::vector<double> input, std::vector<double> expected) {  //  MSE - mean squared error
        double loss{};
        std::vector<double> res;
        res = this->forwardPass(input);
        
        // std::cout<<res.size()<<std::endl;
        // std::cout<<res[0]<<std::endl;

        for(int i=0; i<res.size(); ++i) {
            loss += std::pow(res[i]-expected[i],2)/2;
        }

        return loss;
    }

    void backpropagate(std::vector<double> input, std::vector<double> expected, double learningRate, bool showdata=false) {
        std::vector<double> output = forwardPass(input);
        // double loss = this->loss(output, expected);

        std::vector<std::vector<double>> deltas(layers);

        for(int i=0; i<layers; ++i) {
            deltas[i] = std::vector<double>(layerSize[i], 0);
        }

        for(int i=0; i<output.size(); ++i) {    // because softmax is used instead of ReLU, this layer will be treated differently
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
                sum *= (neurons[L][i].lastSum>0 ? 1:0);
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
                    neurons[L][n].weights[w] -= learningRate * deltas[L][n] * neurons[L-1][w].lastActivation; //subtracting weight gradient
                }
                neurons[L][n].bias -= learningRate * deltas[L][n];  //subtracting bias gradient
            }
        }
    }

    void trainBatch(std::vector<std::vector<std::vector<double>>> batch, double learningRate) {

        std::cout<<"started training.\n ";
        double size = batch.size();
        for(int epoch=0; epoch<size; ++epoch) {
            std::cout<<"epoch: "<<epoch<<std::endl;
            backpropagate(batch[epoch][0], batch[epoch][1], learningRate);
        }
        std::cout<<"Training on batch finished. \n";
    }
};




inline uint32_t _byteswap_uint32 (uint32_t n) {
    return 
    ((n & 0xFF000000) >> 24) |
    ((n & 0x00FF0000) >>  8) |
    ((n & 0x0000FF00) <<  8) |
    ((n & 0x000000FF) << 24);
}

struct NeuralData
{
    std::vector<std::vector<std::vector<double>>> images;
    // std::vector<std::vector<std::vector<uint8_t>>> images_uint8_t;
    std::vector<int> labels;

    int size;

    NeuralData(std::string img, std::string nameLabels) {

        
        uint8_t byte{};
        uint32_t bytes4{};
        uint32_t dim{};
        int dims{};
        //read images
        std::ifstream imgfile(img, std::ios::binary);

        if(imgfile)std::cout<<"found images file.\n";
        imgfile.read(reinterpret_cast<char*>(&bytes4),4); // "magic number"
        
        imgfile.read(reinterpret_cast<char*>(&dim),4);    // first dimension (which means amount of images)
        
        dim = _byteswap_uint32(static_cast<uint64_t>(dim));

        std::cout<<dim<<"\n";

        size = dim;
        
        
        images = std::vector<std::vector<std::vector<double>>>(
            dim, std::vector<std::vector<double>>(28,std::vector<double>(28, 0))
        );

        imgfile.read(reinterpret_cast<char*>(&bytes4),4); //other 2 dimensions (size of image) but they're always 28x28
        imgfile.read(reinterpret_cast<char*>(&bytes4),4);
        
        for(int i=0; i<dim; ++i) {
            for(int y=0; y<28; ++y) {
                for(int x=0; x<28; ++x) {
                    imgfile.read(reinterpret_cast<char*>(&byte),1);
                    //255 - 1
                    //0   - 0
                    images[i][y][x] = (double)byte/255.0;
                }
            }
        }
        
        imgfile.close();

        //read labels
        
        std::ifstream lblfile(nameLabels, std::ios::binary);
        
        if(lblfile)std::cout<<"found labels file.\n";
        
        lblfile.read(reinterpret_cast<char*>(&bytes4),4); // "magic number"
        
        lblfile.read(reinterpret_cast<char*>(&dim),4);    // first (and only) dimension
        
        labels = std::vector<int>(dim,0);
    
        
        for(int i=0; i<dim; ++i) {
            lblfile.read(reinterpret_cast<char*>(&byte),1);
            labels[i] = byte;
        }
        
        lblfile.close();
        
    }

    void display(int index) {
        std::cout<<"\n";
        std::cout<<labels[index];
        std::cout<<"\n";
        for(int y=0; y<28; ++y) {
            for(int x=0; x<28; ++x) {
                if(images[index][y][x]) {
                    if(images[index][y][x] > 0.75) {
                        std::cout<<"##";
                    } else if(images[index][y][x] > 0.5) {
                        std::cout<<"++";
                    } else if(images[index][y][x] > 0.25) {
                        std::cout<<"--";
                    } else {
                        std::cout<<"..";
                    }
                } else std::cout<<"  ";
            } std::cout<<"\n";
        }
        std::cout<<"\n";
        std::cout<<"\n";
    }



    std::vector<double> getX(int i) {
        std::vector<double> X;
        X = std::vector<double>(28*28, 0);
        for(int y=0; y<28; ++y) {
            for(int x=0; x<28; ++x) {
                X[y*28+x] = images[i][y][x];
            }
        }
        return X;
    }

    std::vector<double> getY(int i) {
        std::vector<double> Y(10, 0);
        Y[labels[i]] = 1;
        return Y;
    }

    std::vector<std::vector<std::vector<double>>> getBatch(int istart, int len) {
        // std::cout<<"getting batch";
        std::vector<std::vector<std::vector<double>>> batch(len);
        for(int i=0; i<len; ++i) {
            // std::cout<<i<<"\n";
            std::vector<std::vector<double>> XY(2);
            XY[0] = getX(i+istart);
            XY[1] = getY(i+istart);
            
            batch[i] = XY;
        }
        
        // std::cout<<"end of batch";
        return batch;
    }

};






void testLoss(MultiLayerPerceptron nn, NeuralData data) {
    double losssum{};

    for(int i=0; i<100; ++i) {
        losssum += nn.loss(data.getX(i), data.getY(i));
    }

    std::cout<<"summed loss: "<<losssum<<std::endl;
}


void test() {
    // std::string labelsName = "t10k-labels.idx1-ubyte";
    // std::string imagesName = "t10k-images.idx3-ubyte";
    std::string labelsName = "train-labels-idx1-ubyte";
    std::string imagesName = "train-images-idx3-ubyte";
    
    NeuralData data(imagesName, labelsName);
    

    
    std::vector<int> layers = {28*28,128,10};
    
    MultiLayerPerceptron nn(layers);
    
    
    testLoss(nn,data);
    
    nn.trainBatch(data.getBatch(0,60000), 0.001);
    
    testLoss(nn, data);
    
    // data.display(0);
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
