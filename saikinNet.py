import chainer
import chainer.functions as F
import chainer.links as L
from chainer import cuda
import chainercv
import numpy as np


class SaikinNet(chainer.Chain):
    SRCX = 720
    SRCY = 1280
    DSTX = SRCX
    DSTY = SRCY
    CLASSES = 2

    # Define each layer
    def __init__(self):
        super().__init__()
        with self.init_scope():
            # Convolution layers
            self.conv1 = L.Convolution2D(in_channels=1, out_channels=16, ksize=8, stride=2, pad=4)
            self.conv2 = L.Convolution2D(in_channels=16, out_channels=32, ksize=5, stride=2, pad=2)
            self.conv3 = L.Convolution2D(in_channels=32, out_channels=64, ksize=5, stride=2, pad=2)
            self.conv4 = L.Convolution2D(in_channels=64, out_channels=64, ksize=3, stride=1, pad=1)
            self.conv5 = L.Convolution2D(in_channels=64, out_channels=128, ksize=2, stride=1, pad=0)

            # Batch Normalizations
            self.bn1 = L.BatchNormalization(16)
            self.bn2 = L.BatchNormalization(32)
            self.bn3 = L.BatchNormalization(64)
            self.bn4 = L.BatchNormalization(64)

        self.class_weight = None
        # self.score = None

    # Define connections between layers
    def _logits(self, x):
        # Batch normalization after ReLU as activation function except the final layer
        h = self.bn1(F.relu(self.conv1(x)))
        h = self.bn2(F.relu(self.conv2(h)))
        h = self.bn3(F.relu(self.conv3(h)))
        h = self.bn4(F.relu(self.conv4(h)))
        h = self.conv5(h)
        h = F.depth2space(h, self.DSTX // h.shape[3])  # Resize the final layer to the original image size by Pixel Shuffler method

        return h

    # Run the neural network for training
    def __call__(self, x, t):
        xp = cuda.get_array_module(x)
        h = self._logits(x)  # Run the neural network

        """
        if chainer.config.train:
            class_weight = self.class_weight / self.class_weight.sum()
        else:
            class_weight = xp.array([1., 1.], dtype=self.class_weight.dtype)
        """

        with chainer.using_config('use_cudnn', 'never'):
            loss = F.softmax_cross_entropy(h, t, class_weight=self.class_weight)    # Loss function
        if xp.isnan(loss.data):
            raise RuntimeError("ERROR in SaikinNet: loss.data is nan!")

        label = xp.argmax(h.data, axis=1).astype(xp.int32)  # Convert score images to label images
        result = chainercv.evaluations.eval_semantic_segmentation(chainer.cuda.to_cpu(label),  chainer.cuda.to_cpu(t))  # Evaluate accuracy
        miou = result['miou']
        pacc = result['pixel_accuracy']
        mcacc = result['mean_class_accuracy']
        weight = self.class_weight[1] / self.class_weight[0]

        # Report training parameters and evaluation values every iteration
        chainer.report({
            'loss': loss,
            'miou': miou,
            'miou_error': 1 - miou,
            'pixel_accuracy': pacc,
            'pixel_error': 1 - pacc,
            'class_accuracy': mcacc,
            'class_error': 1 - mcacc,
            'weight': weight,
        }, self)
        """
        chainer.report(
            {'class_weight[{}]'.format(i): w
             for i, w in enumerate(cuda.to_cpu(class_weight).ravel())}, self)
        """
        return loss

    # Run the neural network for inference
    def predict(self, x):
        with chainer.using_config('train', False):
            h = self._logits(x)  # Run the neural network
        with chainer.using_config('use_cudnn', 'never'):
            return F.softmax(h)  # Calculate softmax of score images


def _main():
    pass


if __name__ == '__main__':
    _main()
