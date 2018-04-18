XnorNet-Pytorch
===
This is a Pytorch implementation of the [XNOR-Networks](https://arxiv.org/pdf/1603.05279.pdf) for the MNIST dataset. The model structure is LeNet-5. The dataset is provided by [torchvision](http://pytorch.org/docs/master/torchvision/).
## Requirements
* Python3, Numpy
* Pytorch 0.3.1
## Usage
	$ git clone https://github.com/buaabai/XnorNet
	$ python main.py --epochs 100
You can use
	
	$ python main.py -h
to check other parameters.
## Todo
* LeNet-5 for MNIST
## Notes
In the paper, the gradient in backward after the scaled sign function is inaccurate. The correct equation is [here](https://github.com/jiecaoyu/XNOR-Net-PyTorch/blob/master/notes/notes.pdf).
## How It Works
* `model.py`
This file contains four classes, BinAvtiv, BinConv2d, BinLinear and LeNet5_Bin. BinActiv is used to compute the binary activations and alpha and as a module in BinConv2d and BinLinear. LeNet5_Bin is composed of BinConv2d and BinLinear.And only the activations is binarized now.
* `utils.py`
This file contains binarized operation and update operation for both convolution layer and linear layer on weights. The binarized operation includes clamp, save and binarize(the clamp operation refers to [this papet](https://arxiv.org/pdf/1602.02830.pdf)). The update operation computes the gradients from binarized weights.
* `main.py`
This file is the main function, which includes training, testing and saving the best model. When we are training the network, firstly, we binarize the weights of the model, secondly we use the model with binarized activations and weights, and then we call the backward of loss, lastly we update the weights in binary way.
## Reference
* https://github.com/jiecaoyu/XNOR-Net-PyTorch
* https://github.com/itayhubara/BinaryNet.pytorch
* https://github.com/MatthieuCourbariaux/BinaryNet
* https://github.com/MatthieuCourbariaux/BinaryConnect

	