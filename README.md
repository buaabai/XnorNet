XnorNet-Pytorch
===
This is a Pytorch implementation of the [XNOR-Networks](https://arxiv.org/pdf/1603.05279.pdf) for the MNIST dataset. The model structure is LeNet-5. The dataset is provided by [torchvision](http://pytorch.org/docs/master/torchvision/).
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
This file contains four classes, BinAvtiv, BinConv2d, BinLinear and LeNet5_Bin.BinActiv is used to compute the binary activations and \alpha.
	