
#include <iostream>
#include <vector>

#include "CNN.h"

using namespace std;


bool        adam;
double      bias, eta;
vector<int> image_1{1,28,28}, kernels_1{8,3,3,1};
vector<int> image_2{8,13,13}, kernels_2{2,3,3,8},  hidden{72};
int         input_layer, num_classes, epochs, padding, stride;


int main(int argc, char ** argv){

    //network istantiation
    ofstream file1;
    string data = "cnndata.txt";
    file1.open(data);
    CNN network;

    //build the network 

    network.add_conv(image_1, kernels_1, padding= 0, stride= 2, bias= 0.1, eta= 0.01 );
    network.add_conv(image_2 , kernels_2 , padding= 0, stride= 2, bias= 0.1, eta= 0.01);
    network.add_dense(input_layer=2*6*6, hidden, num_classes=10, bias=1.0,  adam=false, eta=0.5);

    //load the wanted dataset

    network.load_dataset("MNIST");

    //sanity check

    network.sanity_check();

    //train the network (Batch Size = 1)

    network.training(epochs=1, 1);

    //evaluate new samples 

    network.testing(1);
    
    for (int j = 0; j < network._convlist.size(); j++)
    {
        if(j<5000)
            file1 << network._convlist[j] << '\n';

    }

    file1.close();



    return 0;

}
