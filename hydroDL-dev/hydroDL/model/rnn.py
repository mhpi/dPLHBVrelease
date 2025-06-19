import math
import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from .dropout import DropMask, createMask
from . import cnn
import csv
import numpy as np
import random
import importlib

class LSTMcell_untied(torch.nn.Module):
    def __init__(self,
                 *,
                 inputSize,
                 hiddenSize,
                 train=True,
                 dr=0.5,
                 drMethod='gal+sem',
                 gpu=0):
        super(LSTMcell_untied, self).__init__()
        self.inputSize = inputSize
        self.hiddenSize = inputSize
        self.dr = dr

        self.w_xi = Parameter(torch.Tensor(hiddenSize, inputSize))
        self.w_xf = Parameter(torch.Tensor(hiddenSize, inputSize))
        self.w_xo = Parameter(torch.Tensor(hiddenSize, inputSize))
        self.w_xc = Parameter(torch.Tensor(hiddenSize, inputSize))

        self.w_hi = Parameter(torch.Tensor(hiddenSize, hiddenSize))
        self.w_hf = Parameter(torch.Tensor(hiddenSize, hiddenSize))
        self.w_ho = Parameter(torch.Tensor(hiddenSize, hiddenSize))
        self.w_hc = Parameter(torch.Tensor(hiddenSize, hiddenSize))

        self.b_i = Parameter(torch.Tensor(hiddenSize))
        self.b_f = Parameter(torch.Tensor(hiddenSize))
        self.b_o = Parameter(torch.Tensor(hiddenSize))
        self.b_c = Parameter(torch.Tensor(hiddenSize))

        self.drMethod = drMethod.split('+')
        self.gpu = gpu
        self.train = train
        if gpu >= 0:
            self = self.cuda(gpu)
            self.is_cuda = True
        else:
            self.is_cuda = False
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hiddenSize)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def init_mask(self, x, h, c):
        self.maskX_i = createMask(x, self.dr)
        self.maskX_f = createMask(x, self.dr)
        self.maskX_c = createMask(x, self.dr)
        self.maskX_o = createMask(x, self.dr)

        self.maskH_i = createMask(h, self.dr)
        self.maskH_f = createMask(h, self.dr)
        self.maskH_c = createMask(h, self.dr)
        self.maskH_o = createMask(h, self.dr)

        self.maskC = createMask(c, self.dr)

        self.maskW_xi = createMask(self.w_xi, self.dr)
        self.maskW_xf = createMask(self.w_xf, self.dr)
        self.maskW_xc = createMask(self.w_xc, self.dr)
        self.maskW_xo = createMask(self.w_xo, self.dr)
        self.maskW_hi = createMask(self.w_hi, self.dr)
        self.maskW_hf = createMask(self.w_hf, self.dr)
        self.maskW_hc = createMask(self.w_hc, self.dr)
        self.maskW_ho = createMask(self.w_ho, self.dr)

    def forward(self, x, hidden):
        h0, c0 = hidden
        doDrop = self.training and self.dr > 0.0

        if doDrop:
            self.init_mask(x, h0, c0)

        if doDrop and 'drH' in self.drMethod:
            h0_i = h0.mul(self.maskH_i)
            h0_f = h0.mul(self.maskH_f)
            h0_c = h0.mul(self.maskH_c)
            h0_o = h0.mul(self.maskH_o)
        else:
            h0_i = h0
            h0_f = h0
            h0_c = h0
            h0_o = h0

        if doDrop and 'drX' in self.drMethod:
            x_i = x.mul(self.maskX_i)
            x_f = x.mul(self.maskX_f)
            x_c = x.mul(self.maskX_c)
            x_o = x.mul(self.maskX_o)
        else:
            x_i = x
            x_f = x
            x_c = x
            x_o = x

        if doDrop and 'drW' in self.drMethod:
            w_xi = self.w_xi.mul(self.maskW_xi)
            w_xf = self.w_xf.mul(self.maskW_xf)
            w_xc = self.w_xc.mul(self.maskW_xc)
            w_xo = self.w_xo.mul(self.maskW_xo)
            w_hi = self.w_hi.mul(self.maskW_hi)
            w_hf = self.w_hf.mul(self.maskW_hf)
            w_hc = self.w_hc.mul(self.maskW_hc)
            w_ho = self.w_ho.mul(self.maskW_ho)
        else:
            w_xi = self.w_xi
            w_xf = self.w_xf
            w_xc = self.w_xc
            w_xo = self.w_xo
            w_hi = self.w_hi
            w_hf = self.w_hf
            w_hc = self.w_hc
            w_ho = self.w_ho

        gate_i = F.linear(x_i, w_xi) + F.linear(h0_i, w_hi) + self.b_i
        gate_f = F.linear(x_f, w_xf) + F.linear(h0_f, w_hf) + self.b_f
        gate_c = F.linear(x_c, w_xc) + F.linear(h0_c, w_hc) + self.b_c
        gate_o = F.linear(x_o, w_xo) + F.linear(h0_o, w_ho) + self.b_o

        gate_i = F.sigmoid(gate_i)
        gate_f = F.sigmoid(gate_f)
        gate_c = F.tanh(gate_c)
        gate_o = F.sigmoid(gate_o)

        if doDrop and 'drC' in self.drMethod:
            gate_c = gate_c.mul(self.maskC)

        c1 = (gate_f * c0) + (gate_i * gate_c)
        h1 = gate_o * F.tanh(c1)

        return h1, c1


class LSTMcell_tied(torch.nn.Module):
    def __init__(self,
                 *,
                 inputSize,
                 hiddenSize,
                 mode='train',
                 dr=0.5,
                 drMethod='drX+drW+drC',
                 gpu=1):
        super(LSTMcell_tied, self).__init__()

        self.inputSize = inputSize
        self.hiddenSize = hiddenSize
        self.dr = dr

        self.w_ih = Parameter(torch.Tensor(hiddenSize * 4, inputSize))
        self.w_hh = Parameter(torch.Tensor(hiddenSize * 4, hiddenSize))
        self.b_ih = Parameter(torch.Tensor(hiddenSize * 4))
        self.b_hh = Parameter(torch.Tensor(hiddenSize * 4))

        self.drMethod = drMethod.split('+')
        self.gpu = gpu
        self.mode = mode
        if mode == 'train':
            self.train(mode=True)
        elif mode == 'test':
            self.train(mode=False)
        elif mode == 'drMC':
            self.train(mode=False)

        if gpu >= 0:
            self = self.cuda()
            self.is_cuda = True
        else:
            self.is_cuda = False
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hiddenSize)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def reset_mask(self, x, h, c):
        self.maskX = createMask(x, self.dr)
        self.maskH = createMask(h, self.dr)
        self.maskC = createMask(c, self.dr)
        self.maskW_ih = createMask(self.w_ih, self.dr)
        self.maskW_hh = createMask(self.w_hh, self.dr)

    def forward(self, x, hidden, *, resetMask=True, doDropMC=False):
        if self.dr > 0 and (doDropMC is True or self.training is True):
            doDrop = True
        else:
            doDrop = False

        batchSize = x.size(0)
        h0, c0 = hidden
        if h0 is None:
            h0 = x.new_zeros(batchSize, self.hiddenSize, requires_grad=False)
        if c0 is None:
            c0 = x.new_zeros(batchSize, self.hiddenSize, requires_grad=False)

        if self.dr > 0 and self.training is True and resetMask is True:
            self.reset_mask(x, h0, c0)

        if doDrop is True and 'drH' in self.drMethod:
            h0 = DropMask.apply(h0, self.maskH, True)

        if doDrop is True and 'drX' in self.drMethod:
            x = DropMask.apply(x, self.maskX, True)

        if doDrop is True and 'drW' in self.drMethod:
            w_ih = DropMask.apply(self.w_ih, self.maskW_ih, True)
            w_hh = DropMask.apply(self.w_hh, self.maskW_hh, True)
        else:
            # self.w are parameters, while w are not
            w_ih = self.w_ih
            w_hh = self.w_hh

        gates = F.linear(x, w_ih, self.b_ih) + \
            F.linear(h0, w_hh, self.b_hh)
        gate_i, gate_f, gate_c, gate_o = gates.chunk(4, 1)

        gate_i = torch.sigmoid(gate_i)
        gate_f = torch.sigmoid(gate_f)
        gate_c = torch.tanh(gate_c)
        gate_o = torch.sigmoid(gate_o)

        if self.training is True and 'drC' in self.drMethod:
            gate_c = gate_c.mul(self.maskC)

        c1 = (gate_f * c0) + (gate_i * gate_c)
        h1 = gate_o * torch.tanh(c1)

        return h1, c1


class CudnnLstm(torch.nn.Module):
    def __init__(self, *, inputSize, hiddenSize, dr=0.5, drMethod='drW',
                 gpu=0):
        super(CudnnLstm, self).__init__()
        self.inputSize = inputSize
        self.hiddenSize = hiddenSize
        self.dr = dr
        self.w_ih = Parameter(torch.Tensor(hiddenSize * 4, inputSize))
        self.w_hh = Parameter(torch.Tensor(hiddenSize * 4, hiddenSize))
        self.b_ih = Parameter(torch.Tensor(hiddenSize * 4))
        self.b_hh = Parameter(torch.Tensor(hiddenSize * 4))
        self._all_weights = [['w_ih', 'w_hh', 'b_ih', 'b_hh']]
        self.cuda()

        self.reset_mask()
        self.reset_parameters()

    def _apply(self, fn):
        ret = super(CudnnLstm, self)._apply(fn)
        return ret

    def __setstate__(self, d):
        super(CudnnLstm, self).__setstate__(d)
        self.__dict__.setdefault('_data_ptrs', [])
        if 'all_weights' in d:
            self._all_weights = d['all_weights']
        if isinstance(self._all_weights[0][0], str):
            return
        self._all_weights = [['w_ih', 'w_hh', 'b_ih', 'b_hh']]

    def reset_mask(self):
        self.maskW_ih = createMask(self.w_ih, self.dr)
        self.maskW_hh = createMask(self.w_hh, self.dr)

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hiddenSize)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)


    def forward(self, input, hx=None, cx=None, doDropMC=False, dropoutFalse=False):
        # dropoutFalse: it will ensure doDrop is false, unless doDropMC is true
        if dropoutFalse and (not doDropMC):
            doDrop = False
        elif self.dr > 0 and (doDropMC is True or self.training is True):
            doDrop = True
        else:
            doDrop = False

        batchSize = input.size(1)

        if hx is None:
            hx = input.new_zeros(
                1, batchSize, self.hiddenSize, requires_grad=False)
        if cx is None:
            cx = input.new_zeros(
                1, batchSize, self.hiddenSize, requires_grad=False)

        # cuDNN backend - disabled flat weight
        # handle = torch.backends.cudnn.get_handle()
        if doDrop is True:
            self.reset_mask()
            weight = [
                DropMask.apply(self.w_ih, self.maskW_ih, True),
                DropMask.apply(self.w_hh, self.maskW_hh, True), self.b_ih,
                self.b_hh
            ]
        else:
            weight = [self.w_ih, self.w_hh, self.b_ih, self.b_hh]

        # output, hy, cy, reserve, new_weight_buf = torch._cudnn_rnn(
        #     input, weight, 4, None, hx, cx, torch.backends.cudnn.CUDNN_LSTM,
        #     self.hiddenSize, 1, False, 0, self.training, False, (), None)
        if torch.__version__ < "1.8":
            output, hy, cy, reserve, new_weight_buf = torch._cudnn_rnn(
                input, weight, 4, None, hx, cx, 2,  # 2 means LSTM
                self.hiddenSize, 1, False, 0, self.training, False, (), None)
        else:
            output, hy, cy, reserve, new_weight_buf = torch._cudnn_rnn(
                input, weight, 4, None, hx, cx, 2,  # 2 means LSTM
                self.hiddenSize, 0, 1, False, 0, self.training, False, (), None)
        return output, (hy, cy)

    @property
    def all_weights(self):
        return [[getattr(self, weight) for weight in weights]
                for weights in self._all_weights]

class CNN1dkernel(torch.nn.Module):
    def __init__(self,
                 *,
                 ninchannel=1,
                 nkernel=3,
                 kernelSize=3,
                 stride=1,
                 padding=0):
        super(CNN1dkernel, self).__init__()
        self.cnn1d = torch.nn.Conv1d(
            in_channels=ninchannel,
            out_channels=nkernel,
            kernel_size=kernelSize,
            padding=padding,
            stride=stride,
        )

    def forward(self, x):
        output = F.relu(self.cnn1d(x))
        return output

class CudnnLstmModel(torch.nn.Module):
    def __init__(self, *, nx, ny, hiddenSize, dr=0.5):
        super(CudnnLstmModel, self).__init__()
        self.nx = nx
        self.ny = ny
        self.hiddenSize = hiddenSize
        self.ct = 0
        self.nLayer = 1
        self.linearIn = torch.nn.Linear(nx, hiddenSize)
        self.lstm = CudnnLstm(
            inputSize=hiddenSize, hiddenSize=hiddenSize, dr=dr)
        self.linearOut = torch.nn.Linear(hiddenSize, ny)
        self.gpu = 1
        # self.drtest = torch.nn.Dropout(p=0.4)

    def forward(self, x, doDropMC=False, dropoutFalse=False):
        x0 = F.relu(self.linearIn(x))
        outLSTM, (hn, cn) = self.lstm(x0, doDropMC=doDropMC, dropoutFalse=dropoutFalse)
        # outLSTMdr = self.drtest(outLSTM)
        out = self.linearOut(outLSTM)
        return out


class CNN1dLSTMmodel(torch.nn.Module):
    def __init__(self, *, nx, ny, nobs, hiddenSize,
                 nkernel=(10,5), kernelSize=(3,3), stride=(2,1), dr=0.5, poolOpt=None):
        # two convolutional layer
        super(CNN1dLSTMmodel, self).__init__()
        self.nx = nx
        self.ny = ny
        self.obs = nobs
        self.hiddenSize = hiddenSize
        nlayer = len(nkernel)
        self.features = nn.Sequential()
        ninchan = 1
        Lout = nobs
        for ii in range(nlayer):
            ConvLayer = CNN1dkernel(
                ninchannel=ninchan, nkernel=nkernel[ii], kernelSize=kernelSize[ii], stride=stride[ii])
            self.features.add_module('CnnLayer%d' % (ii + 1), ConvLayer)
            ninchan = nkernel[ii]
            Lout = cnn.calConvSize(lin=Lout, kernel=kernelSize[ii], stride=stride[ii])
            self.features.add_module('Relu%d' % (ii + 1), nn.ReLU())
            if poolOpt is not None:
                self.features.add_module('Pooling%d' % (ii + 1), nn.MaxPool1d(poolOpt[ii]))
                Lout = cnn.calPoolSize(lin=Lout, kernel=poolOpt[ii])
        self.Ncnnout = int(Lout*nkernel[-1]) # total CNN feature number after convolution
        Nf = self.Ncnnout + nx
        self.linearIn = torch.nn.Linear(Nf, hiddenSize)
        self.lstm = CudnnLstm(
            inputSize=hiddenSize, hiddenSize=hiddenSize, dr=dr)
        self.linearOut = torch.nn.Linear(hiddenSize, ny)
        self.gpu = 1

    def forward(self, x, z, doDropMC=False):
        nt, ngrid, nobs = z.shape
        z = z.view(nt*ngrid, 1, nobs)
        z0 = self.features(z)
        # z0 = (ntime*ngrid) * nkernel * sizeafterconv
        z0 = z0.view(nt, ngrid, self.Ncnnout)
        x0 = torch.cat((x, z0), dim=2)
        x0 = F.relu(self.linearIn(x0))
        outLSTM, (hn, cn) = self.lstm(x0, doDropMC=doDropMC)
        out = self.linearOut(outLSTM)
        # out = rho/time * batchsize * Ntargetvar
        return out

class CNN1dLSTMInmodel(torch.nn.Module):
    # Directly add the CNN extracted features into LSTM inputSize
    def __init__(self, *, nx, ny, nobs, hiddenSize,
                 nkernel=(10,5), kernelSize=(3,3), stride=(2,1), dr=0.5, poolOpt=None, cnndr=0.0):
        # two convolutional layer
        super(CNN1dLSTMInmodel, self).__init__()
        self.nx = nx
        self.ny = ny
        self.obs = nobs
        self.hiddenSize = hiddenSize
        nlayer = len(nkernel)
        self.features = nn.Sequential()
        ninchan = 1
        Lout = nobs
        for ii in range(nlayer):
            ConvLayer = CNN1dkernel(
                ninchannel=ninchan, nkernel=nkernel[ii], kernelSize=kernelSize[ii], stride=stride[ii])
            self.features.add_module('CnnLayer%d' % (ii + 1), ConvLayer)
            if cnndr != 0.0:
                self.features.add_module('dropout%d' % (ii + 1), nn.Dropout(p=cnndr))
            ninchan = nkernel[ii]
            Lout = cnn.calConvSize(lin=Lout, kernel=kernelSize[ii], stride=stride[ii])
            self.features.add_module('Relu%d' % (ii + 1), nn.ReLU())
            if poolOpt is not None:
                self.features.add_module('Pooling%d' % (ii + 1), nn.MaxPool1d(poolOpt[ii]))
                Lout = cnn.calPoolSize(lin=Lout, kernel=poolOpt[ii])
        self.Ncnnout = int(Lout*nkernel[-1]) # total CNN feature number after convolution
        Nf = self.Ncnnout + hiddenSize
        self.linearIn = torch.nn.Linear(nx, hiddenSize)
        self.lstm = CudnnLstm(
            inputSize=Nf, hiddenSize=hiddenSize, dr=dr)
        self.linearOut = torch.nn.Linear(hiddenSize, ny)
        self.gpu = 1

    def forward(self, x, z, doDropMC=False):
        nt, ngrid, nobs = z.shape
        z = z.view(nt*ngrid, 1, nobs)
        z0 = self.features(z)
        # z0 = (ntime*ngrid) * nkernel * sizeafterconv
        z0 = z0.view(nt, ngrid, self.Ncnnout)
        x = F.relu(self.linearIn(x))
        x0 = torch.cat((x, z0), dim=2)
        outLSTM, (hn, cn) = self.lstm(x0, doDropMC=doDropMC)
        out = self.linearOut(outLSTM)
        # out = rho/time * batchsize * Ntargetvar
        return out

class CNN1dLCmodel(torch.nn.Module):
    # add the CNN extracted features into original LSTM input, then pass through linear layer
    def __init__(self, *, nx, ny, nobs, hiddenSize,
                 nkernel=(10,5), kernelSize=(3,3), stride=(2,1), dr=0.5, poolOpt=None, cnndr=0.0):
        # two convolutional layer
        super(CNN1dLCmodel, self).__init__()
        self.nx = nx
        self.ny = ny
        self.obs = nobs
        self.hiddenSize = hiddenSize
        nlayer = len(nkernel)
        self.features = nn.Sequential()
        ninchan = 1 # need to modify the hardcode: 4 for smap and 1 for FDC
        Lout = nobs
        for ii in range(nlayer):
            ConvLayer = CNN1dkernel(
                ninchannel=ninchan, nkernel=nkernel[ii], kernelSize=kernelSize[ii], stride=stride[ii])
            self.features.add_module('CnnLayer%d' % (ii + 1), ConvLayer)
            if cnndr != 0.0:
                self.features.add_module('dropout%d' % (ii + 1), nn.Dropout(p=cnndr))
            ninchan = nkernel[ii]
            Lout = cnn.calConvSize(lin=Lout, kernel=kernelSize[ii], stride=stride[ii])
            self.features.add_module('Relu%d' % (ii + 1), nn.ReLU())
            if poolOpt is not None:
                self.features.add_module('Pooling%d' % (ii + 1), nn.MaxPool1d(poolOpt[ii]))
                Lout = cnn.calPoolSize(lin=Lout, kernel=poolOpt[ii])
        self.Ncnnout = int(Lout*nkernel[-1]) # total CNN feature number after convolution
        Nf = self.Ncnnout + nx
        self.linearIn = torch.nn.Linear(Nf, hiddenSize)
        self.lstm = CudnnLstm(
            inputSize=hiddenSize, hiddenSize=hiddenSize, dr=dr)
        self.linearOut = torch.nn.Linear(hiddenSize, ny)
        self.gpu = 1

    def forward(self, x, z, doDropMC=False):
        # z = ngrid*nVar add a channel dimension
        ngrid = z.shape[0]
        rho, BS, Nvar = x.shape
        if len(z.shape) == 2: # for FDC, else 3 dimension for smap
            z = torch.unsqueeze(z, dim=1)
        z0 = self.features(z)
        # z0 = (ngrid) * nkernel * sizeafterconv
        z0 = z0.view(ngrid, self.Ncnnout).repeat(rho,1,1)
        x = torch.cat((x, z0), dim=2)
        x0 = F.relu(self.linearIn(x))
        outLSTM, (hn, cn) = self.lstm(x0, doDropMC=doDropMC)
        out = self.linearOut(outLSTM)
        # out = rho/time * batchsize * Ntargetvar
        return out

class CNN1dLCInmodel(torch.nn.Module):
    # Directly add the CNN extracted features into LSTM inputSize
    def __init__(self, *, nx, ny, nobs, hiddenSize,
                 nkernel=(10,5), kernelSize=(3,3), stride=(2,1), dr=0.5, poolOpt=None, cnndr=0.0):
        # two convolutional layer
        super(CNN1dLCInmodel, self).__init__()
        self.nx = nx
        self.ny = ny
        self.obs = nobs
        self.hiddenSize = hiddenSize
        nlayer = len(nkernel)
        self.features = nn.Sequential()
        ninchan = 1
        Lout = nobs
        for ii in range(nlayer):
            ConvLayer = CNN1dkernel(
                ninchannel=ninchan, nkernel=nkernel[ii], kernelSize=kernelSize[ii], stride=stride[ii])
            self.features.add_module('CnnLayer%d' % (ii + 1), ConvLayer)
            if cnndr != 0.0:
                self.features.add_module('dropout%d' % (ii + 1), nn.Dropout(p=cnndr))
            ninchan = nkernel[ii]
            Lout = cnn.calConvSize(lin=Lout, kernel=kernelSize[ii], stride=stride[ii])
            self.features.add_module('Relu%d' % (ii + 1), nn.ReLU())
            if poolOpt is not None:
                self.features.add_module('Pooling%d' % (ii + 1), nn.MaxPool1d(poolOpt[ii]))
                Lout = cnn.calPoolSize(lin=Lout, kernel=poolOpt[ii])
        self.Ncnnout = int(Lout*nkernel[-1]) # total CNN feature number after convolution
        Nf = self.Ncnnout + hiddenSize
        self.linearIn = torch.nn.Linear(nx, hiddenSize)
        self.lstm = CudnnLstm(
            inputSize=Nf, hiddenSize=hiddenSize, dr=dr)
        self.linearOut = torch.nn.Linear(hiddenSize, ny)
        self.gpu = 1

    def forward(self, x, z, doDropMC=False):
        # z = ngrid*nVar add a channel dimension
        ngrid, nobs = z.shape
        rho, BS, Nvar = x.shape
        z = torch.unsqueeze(z, dim=1)
        z0 = self.features(z)
        # z0 = (ngrid) * nkernel * sizeafterconv
        z0 = z0.view(ngrid, self.Ncnnout).repeat(rho,1,1)
        x = F.relu(self.linearIn(x))
        x0 = torch.cat((x, z0), dim=2)
        outLSTM, (hn, cn) = self.lstm(x0, doDropMC=doDropMC)
        out = self.linearOut(outLSTM)
        # out = rho/time * batchsize * Ntargetvar
        return out

class CudnnInvLstmModel(torch.nn.Module):
    # using cudnnLstm to extract features from SMAP observations
    def __init__(self, *, nx, ny, hiddenSize, ninv, nfea, hiddeninv, dr=0.5, drinv=0.5):
        # two LSTM
        super(CudnnInvLstmModel, self).__init__()
        self.nx = nx
        self.ny = ny
        self.hiddenSize = hiddenSize
        self.ninv = ninv
        self.nfea = nfea
        self.hiddeninv = hiddeninv
        self.lstminv = CudnnLstmModel(
            nx=ninv, ny=nfea, hiddenSize=hiddeninv, dr=drinv)
        self.lstm = CudnnLstmModel(
            nx=nfea+nx, ny=ny, hiddenSize=hiddenSize, dr=dr)
        self.gpu = 1

    def forward(self, x, z, doDropMC=False):
        Gen = self.lstminv(z)
        dim = x.shape;
        nt = dim[0]
        invpara = Gen[-1, :, :].repeat(nt, 1, 1)
        x1 = torch.cat((x, invpara), dim=2)
        out = self.lstm(x1)
        # out = rho/time * batchsize * Ntargetvar
        return out


class LstmCloseModel(torch.nn.Module):
    def __init__(self, *, nx, ny, hiddenSize, dr=0.5, fillObs=True):
        super(LstmCloseModel, self).__init__()
        self.nx = nx
        self.ny = ny
        self.hiddenSize = hiddenSize
        self.ct = 0
        self.nLayer = 1
        self.linearIn = torch.nn.Linear(nx + 1, hiddenSize)
        # self.lstm = CudnnLstm(
        #     inputSize=hiddenSize, hiddenSize=hiddenSize, dr=dr)
        self.lstm = LSTMcell_tied(
            inputSize=hiddenSize, hiddenSize=hiddenSize, dr=dr, drMethod='drW')
        self.linearOut = torch.nn.Linear(hiddenSize, ny)
        self.gpu = 1
        self.fillObs = fillObs

    def forward(self, x, y=None):
        nt, ngrid, nx = x.shape
        yt = torch.zeros(ngrid, 1).cuda()
        out = torch.zeros(nt, ngrid, self.ny).cuda()
        ht = None
        ct = None
        resetMask = True
        for t in range(nt):
            if self.fillObs is True:
                ytObs = y[t, :, :]
                mask = ytObs == ytObs
                yt[mask] = ytObs[mask]
            xt = torch.cat((x[t, :, :], yt), 1)
            x0 = F.relu(self.linearIn(xt))
            ht, ct = self.lstm(x0, hidden=(ht, ct), resetMask=resetMask)
            yt = self.linearOut(ht)
            resetMask = False
            out[t, :, :] = yt
        return out


class AnnModel(torch.nn.Module):
    def __init__(self, *, nx, ny, hiddenSize):
        super(AnnModel, self).__init__()
        self.hiddenSize = hiddenSize
        self.i2h = nn.Linear(nx, hiddenSize)
        self.h2h = nn.Linear(hiddenSize, hiddenSize)
        self.h2o = nn.Linear(hiddenSize, ny)
        self.ny = ny

    def forward(self, x, y=None):
        nt, ngrid, nx = x.shape
        yt = torch.zeros(ngrid, 1).cuda()
        out = torch.zeros(nt, ngrid, self.ny).cuda()
        for t in range(nt):
            xt = x[t, :, :]
            ht = F.relu(self.i2h(xt))
            ht2 = self.h2h(ht)
            yt = self.h2o(ht2)
            out[t, :, :] = yt
        return out


class AnnCloseModel(torch.nn.Module):
    def __init__(self, *, nx, ny, hiddenSize, fillObs=True):
        super(AnnCloseModel, self).__init__()
        self.hiddenSize = hiddenSize
        self.i2h = nn.Linear(nx + 1, hiddenSize)
        self.h2h = nn.Linear(hiddenSize, hiddenSize)
        self.h2o = nn.Linear(hiddenSize, ny)
        self.fillObs = fillObs
        self.ny = ny

    def forward(self, x, y=None):
        nt, ngrid, nx = x.shape
        yt = torch.zeros(ngrid, 1).cuda()
        out = torch.zeros(nt, ngrid, self.ny).cuda()
        for t in range(nt):
            if self.fillObs is True:
                ytObs = y[t, :, :]
                mask = ytObs == ytObs
                yt[mask] = ytObs[mask]
            xt = torch.cat((x[t, :, :], yt), 1)
            ht = F.relu(self.i2h(xt))
            ht2 = self.h2h(ht)
            yt = self.h2o(ht2)
            out[t, :, :] = yt
        return out


class LstmCnnCond(nn.Module):
    def __init__(self,
                 *,
                 nx,
                 ny,
                 ct,
                 opt=1,
                 hiddenSize=64,
                 cnnSize=32,
                 cp1=(64, 3, 2),
                 cp2=(128, 5, 2),
                 dr=0.5):
        super(LstmCnnCond, self).__init__()

        # opt == 1: cnn output as initial state of LSTM (h0)
        # opt == 2: cnn output as additional output of LSTM
        # opt == 3: cnn output as constant input of LSTM

        if opt == 1:
            cnnSize = hiddenSize

        self.nx = nx
        self.ny = ny
        self.ct = ct
        self.ctRm = False
        self.hiddenSize = hiddenSize
        self.opt = opt

        self.cnn = cnn.Cnn1d(nx=nx, nt=ct, cnnSize=cnnSize, cp1=cp1, cp2=cp2)

        self.lstm = CudnnLstm(
            inputSize=hiddenSize, hiddenSize=hiddenSize, dr=dr)
        if opt == 3:
            self.linearIn = torch.nn.Linear(nx + cnnSize, hiddenSize)
        else:
            self.linearIn = torch.nn.Linear(nx, hiddenSize)
        if opt == 2:
            self.linearOut = torch.nn.Linear(hiddenSize + cnnSize, ny)
        else:
            self.linearOut = torch.nn.Linear(hiddenSize, ny)

    def forward(self, x, xc):
        # x- [nt,ngrid,nx]
        x1 = xc
        x1 = self.cnn(x1)
        x2 = x
        if self.opt == 1:
            x2 = F.relu(self.linearIn(x2))
            x2, (hn, cn) = self.lstm(x2, hx=x1[None, :, :])
            x2 = self.linearOut(x2)
        elif self.opt == 2:
            x1 = x1[None, :, :].repeat(x2.shape[0], 1, 1)
            x2 = F.relu(self.linearIn(x2))
            x2, (hn, cn) = self.lstm(x2)
            x2 = self.linearOut(torch.cat([x2, x1], 2))
        elif self.opt == 3:
            x1 = x1[None, :, :].repeat(x2.shape[0], 1, 1)
            x2 = torch.cat([x2, x1], 2)
            x2 = F.relu(self.linearIn(x2))
            x2, (hn, cn) = self.lstm(x2)
            x2 = self.linearOut(x2)

        return x2


class LstmCnnForcast(nn.Module):
    def __init__(self,
                 *,
                 nx,
                 ny,
                 ct,
                 opt=1,
                 hiddenSize=64,
                 cnnSize=32,
                 cp1=(64, 3, 2),
                 cp2=(128, 5, 2),
                 dr=0.5):
        super(LstmCnnForcast, self).__init__()

        if opt == 1:
            cnnSize = hiddenSize

        self.nx = nx
        self.ny = ny
        self.ct = ct
        self.ctRm = True
        self.hiddenSize = hiddenSize
        self.opt = opt
        self.cnnSize = cnnSize

        if opt == 1:
            self.cnn = cnn.Cnn1d(
                nx=nx + 1, nt=ct, cnnSize=cnnSize, cp1=cp1, cp2=cp2)
        if opt == 2:
            self.cnn = cnn.Cnn1d(
                nx=1, nt=ct, cnnSize=cnnSize, cp1=cp1, cp2=cp2)

        self.lstm = CudnnLstm(
            inputSize=hiddenSize, hiddenSize=hiddenSize, dr=dr)
        self.linearIn = torch.nn.Linear(nx + cnnSize, hiddenSize)
        self.linearOut = torch.nn.Linear(hiddenSize, ny)

    def forward(self, x, y):
        # x- [nt,ngrid,nx]
        nt, ngrid, nx = x.shape
        ct = self.ct
        pt = nt - ct

        if self.opt == 1:
            x1 = torch.cat((y, x), dim=2)
        elif self.opt == 2:
            x1 = y

        x1out = torch.zeros([pt, ngrid, self.cnnSize]).cuda()
        for k in range(pt):
            x1out[k, :, :] = self.cnn(x1[k:k + ct, :, :])

        x2 = x[ct:nt, :, :]
        x2 = torch.cat([x2, x1out], 2)
        x2 = F.relu(self.linearIn(x2))
        x2, (hn, cn) = self.lstm(x2)
        x2 = self.linearOut(x2)

        return x2

class CudnnLstmModel_R2P(torch.nn.Module):
    pass

class CpuLstmModel(torch.nn.Module):
    def __init__(self, *, nx, ny, hiddenSize, dr=0.5):
        super(CpuLstmModel, self).__init__()
        self.nx = nx
        self.ny = ny
        self.hiddenSize = hiddenSize
        self.ct = 0
        self.nLayer = 1
        self.linearIn = torch.nn.Linear(nx, hiddenSize)
        self.lstm = LSTMcell_tied(
            inputSize=hiddenSize, hiddenSize=hiddenSize, dr=dr, drMethod='drW', gpu=-1)
        self.linearOut = torch.nn.Linear(hiddenSize, ny)
        self.gpu = -1

    def forward(self, x, doDropMC=False):
        # x0 = F.relu(self.linearIn(x))
        # outLSTM, (hn, cn) = self.lstm(x0, doDropMC=doDropMC)
        # out = self.linearOut(outLSTM)
        # return out
        nt, ngrid, nx = x.shape
        yt = torch.zeros(ngrid, 1)
        out = torch.zeros(nt, ngrid, self.ny)
        ht = None
        ct = None
        resetMask = True
        for t in range(nt):
            xt = x[t, :, :]
            x0 = F.relu(self.linearIn(xt))
            ht, ct = self.lstm(x0, hidden=(ht, ct), resetMask=resetMask)
            yt = self.linearOut(ht)
            resetMask = False
            out[t, :, :] = yt
        return out


def UH_conv(x,UH,viewmode=1):
    # UH is a vector indicating the unit hydrograph
    # the convolved dimension will be the last dimension
    # UH convolution is
    # Q(t)=\integral(x(\tao)*UH(t-\tao))d\tao
    # conv1d does \integral(w(\tao)*x(t+\tao))d\tao
    # hence we flip the UH
    # https://programmer.group/pytorch-learning-conv1d-conv2d-and-conv3d.html
    # view
    # x: [batch, var, time]
    # UH:[batch, var, uhLen]
    # batch needs to be accommodated by channels and we make use of groups
    # https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
    # https://pytorch.org/docs/stable/nn.functional.html

    mm= x.shape; nb=mm[0]
    m = UH.shape[-1]
    padd = m-1
    if viewmode==1:
        xx = x.view([1,nb,mm[-1]])
        w  = UH.view([nb,1,m])
        groups = nb

    y = F.conv1d(xx, torch.flip(w,[2]), groups=groups, padding=padd, stride=1, bias=None)
    y=y[:,:,0:-padd]
    return y.view(mm)


def UH_gamma(a,b,lenF=10):
    # UH. a [time (same all time steps), batch, var]
    m = a.shape
    w = torch.zeros([lenF, m[1],m[2]])
    aa = F.relu(a[0:lenF,:,:]).view([lenF, m[1],m[2]])+0.1 # minimum 0.1. First dimension of a is repeat
    theta = F.relu(b[0:lenF,:,:]).view([lenF, m[1],m[2]])+0.5 # minimum 0.5
    t = torch.arange(0.5,lenF*1.0).view([lenF,1,1]).repeat([1,m[1],m[2]])
    t = t.cuda(aa.device)
    denom = (aa.lgamma().exp())*(theta**aa)
    mid= t**(aa-1)
    right=torch.exp(-t/theta)
    w = 1/denom*mid*right
    w = w/w.sum(0) # scale to 1 for each UH

    return w

class SimpAnn(torch.nn.Module):
    def __init__(self, *, nx, ny, hiddenSize):
        super(SimpAnn, self).__init__()
        self.hiddenSize = hiddenSize
        self.i2h = nn.Linear(nx, hiddenSize)
        self.h2h = nn.Linear(hiddenSize, hiddenSize)
        self.h2o = nn.Linear(hiddenSize, ny)
        self.ny = ny

    def forward(self, x):
        ht = F.relu(self.i2h(x))
        ht2 = F.relu(self.h2h(ht))
        out = F.relu(self.h2o(ht2))
        return out


class HBVMul(torch.nn.Module):
    """Multi-component HBV model implemented in PyTorch by Dapeng Feng"""

    def __init__(self):
        """Initiate a HBV instance"""
        super(HBVMul, self).__init__()

    def forward(self, x, parameters, mu, muwts, rtwts, bufftime=0, outstate=False, routOpt=False, comprout=False,
                corrwts=None, pcorr=None):
        # Modified from the original numpy version from Beck et al., 2020. (http://www.gloh2o.org/hbv/) which
        # runs the HBV-light hydrological model (Seibert, 2005).
        # NaN values have to be removed from the inputs.
        #
        # Input:
        #     X: dim=[time, basin, var] forcing array with var P(mm/d), T(deg C), PET(mm/d)
        #     parameters: array with parameter values having the following structure and scales:
        #         BETA[1,6]; FC[50,1000]; K0[0.05,0.9]; K1[0.01,0.5]; K2[0.001,0.2]; LP[0.2,1];
        #         PERC[0,10]; UZL[0,100]; TT[-2.5,2.5]; CFMAX[0.5,10]; CFR[0,0.1]; CWH[0,0.2]
        #     mu:number of components; muwts: weights of components if True; rtwts: routing parameters;
        #     bufftime:warm up period; outstate: output state var; routOpt:routing option; comprout:component routing opt
        #     corrwts:P correction weights; pcorr: P correction opt
        #
        #
        # Output, all in mm:
        #     outstate True: output most state variables for warm-up
        #      Qs:simulated streamflow; SNOWPACK:snow depth; MELTWATER:snow water holding depth;
        #      SM:soil storage; SUZ:upper zone storage; SLZ:lower zone storage
        #     outstate False: output the simulated flux array Qall contains
        #      Qs:simulated streamflow=Q0+Q1+Q2; Qsimave0:Q0 component; Qsimave1:Q1 component; Qsimave2:Q2 baseflow componnet
        #      ETave: actual ET

        PRECS = 1e-5 # keep the numerical calculation stable

        # Initialization for warm up states
        if bufftime > 0:
            with torch.no_grad():
                xinit = x[0:bufftime, :, :]
                initmodel = HBVMul()
                Qsinit, SNOWPACK, MELTWATER, SM, SUZ, SLZ = initmodel(xinit, parameters, mu, muwts, rtwts,
                                                                      bufftime=0, outstate=True, routOpt=False, comprout=False,
                                                                      corrwts=corrwts, pcorr=pcorr)
        else:
            # Without warm-up bufftime=0, initialize state variables with zeros
            Ngrid = x.shape[1]
            SNOWPACK = (torch.zeros([Ngrid,mu], dtype=torch.float32) + 0.001).cuda()
            MELTWATER = (torch.zeros([Ngrid,mu], dtype=torch.float32) + 0.001).cuda()
            SM = (torch.zeros([Ngrid,mu], dtype=torch.float32) + 0.001).cuda()
            SUZ = (torch.zeros([Ngrid,mu], dtype=torch.float32) + 0.001).cuda()
            SLZ = (torch.zeros([Ngrid,mu], dtype=torch.float32) + 0.001).cuda()
            # ETact = (torch.zeros([Ngrid,mu], dtype=torch.float32) + 0.001).cuda()

        P = x[bufftime:, :, 0]
        Nstep, Ngrid = P.size()
        if pcorr is not None:
            parPCORR = pcorr[0] + corrwts[:,0]*(pcorr[1]-pcorr[0])
            P = parPCORR.unsqueeze(0).repeat(Nstep, 1) * P
            # print('P corrected')

        Pm= P.unsqueeze(2).repeat(1,1,mu) # precip
        T = x[bufftime:, :, 1]
        Tm = T.unsqueeze(2).repeat(1,1,mu) # temperature
        ETpot = x[bufftime:, :, 2]
        ETpm = ETpot.unsqueeze(2).repeat(1,1,mu) # potential ET


        ## scale the parameters to real vales
        parascaLst = [[1,6], [50,1000], [0.05,0.9], [0.01,0.5], [0.001,0.2], [0.2,1],
                        [0,10], [0,100], [-2.5,2.5], [0.5,10], [0,0.1], [0,0.2]] # HBV para
        routscaLst = [[0,2.9], [0,6.5]] # routing para
        # dim of each para is [Nbasin*Ncomponent]
        parBETA = parascaLst[0][0] + parameters[:,0,:]*(parascaLst[0][1]-parascaLst[0][0])
        parFC = parascaLst[1][0] + parameters[:,1,:]*(parascaLst[1][1]-parascaLst[1][0])
        parK0 = parascaLst[2][0] + parameters[:,2,:]*(parascaLst[2][1]-parascaLst[2][0])
        parK1 = parascaLst[3][0] + parameters[:,3,:]*(parascaLst[3][1]-parascaLst[3][0])
        parK2 = parascaLst[4][0] + parameters[:,4,:]*(parascaLst[4][1]-parascaLst[4][0])
        parLP = parascaLst[5][0] + parameters[:,5,:]*(parascaLst[5][1]-parascaLst[5][0])
        parPERC = parascaLst[6][0] + parameters[:,6,:]*(parascaLst[6][1]-parascaLst[6][0])
        parUZL = parascaLst[7][0] + parameters[:,7,:]*(parascaLst[7][1]-parascaLst[7][0])
        parTT = parascaLst[8][0] + parameters[:,8,:]*(parascaLst[8][1]-parascaLst[8][0])
        parCFMAX = parascaLst[9][0] + parameters[:,9,:]*(parascaLst[9][1]-parascaLst[9][0])
        parCFR = parascaLst[10][0] + parameters[:,10,:]*(parascaLst[10][1]-parascaLst[10][0])
        parCWH = parascaLst[11][0] + parameters[:,11,:]*(parascaLst[11][1]-parascaLst[11][0])

        # Initialize time series of model variables
        Qsimmu = (torch.zeros(Pm.size(), dtype=torch.float32) + 0.001).cuda()
        ETmu = (torch.zeros(Pm.size(), dtype=torch.float32) + 0.001).cuda()

        # Output the three simulated components of total Q
        Qsimmu0 = (torch.zeros(Pm.size(), dtype=torch.float32) + 0.001).cuda()
        Qsimmu1 = (torch.zeros(Pm.size(), dtype=torch.float32) + 0.001).cuda()
        Qsimmu2 = (torch.zeros(Pm.size(), dtype=torch.float32) + 0.001).cuda()


        for t in range(Nstep):
            # Separate precipitation into liquid and solid components
            PRECIP = Pm[t, :, :]  # need to check later, seems repeating with line 52
            RAIN = torch.mul(PRECIP, (Tm[t, :, :] >= parTT).type(torch.float32))
            SNOW = torch.mul(PRECIP, (Tm[t, :, :] < parTT).type(torch.float32))

            # Snow
            SNOWPACK = SNOWPACK + SNOW
            melt = parCFMAX * (Tm[t, :, :] - parTT)
            melt = torch.clamp(melt, min=0.0)
            melt = torch.min(melt, SNOWPACK)
            MELTWATER = MELTWATER + melt
            SNOWPACK = SNOWPACK - melt
            refreezing = parCFR * parCFMAX * (parTT - Tm[t, :, :])
            refreezing = torch.clamp(refreezing, min=0.0)
            refreezing = torch.min(refreezing, MELTWATER)
            SNOWPACK = SNOWPACK + refreezing
            MELTWATER = MELTWATER - refreezing
            tosoil = MELTWATER - (parCWH * SNOWPACK)
            tosoil = torch.clamp(tosoil, min=0.0)
            MELTWATER = MELTWATER - tosoil

            # Soil and evaporation
            soil_wetness = (SM / parFC) ** parBETA
            soil_wetness = torch.clamp(soil_wetness, min=0.0, max=1.0)
            recharge = (RAIN + tosoil) * soil_wetness

            SM = SM + RAIN + tosoil - recharge
            excess = SM - parFC
            excess = torch.clamp(excess, min=0.0)
            SM = SM - excess
            evapfactor = SM / (parLP * parFC)
            evapfactor  = torch.clamp(evapfactor, min=0.0, max=1.0)
            ETact = ETpm[t, :, :] * evapfactor
            ETact = torch.min(SM, ETact)
            SM = torch.clamp(SM - ETact, min=PRECS) # SM can not be zero for gradient tracking
            ETmu[t, :, :] = ETact

            # Groundwater boxes
            SUZ = SUZ + recharge + excess
            PERC = torch.min(SUZ, parPERC)
            SUZ = SUZ - PERC
            Q0 = parK0 * torch.clamp(SUZ - parUZL, min=0.0)
            SUZ = SUZ - Q0
            Q1 = parK1 * SUZ
            SUZ = SUZ - Q1
            SLZ = SLZ + PERC
            Q2 = parK2 * SLZ
            SLZ = SLZ - Q2
            Qsimmu[t, :, :] = Q0 + Q1 + Q2

            # save components
            Qsimmu0[t, :, :] = Q0
            Qsimmu1[t, :, :] = Q1
            Qsimmu2[t, :, :] = Q2


        Qsimave0 = Qsimmu0.mean(-1, keepdim=True)
        Qsimave1 = Qsimmu1.mean(-1, keepdim=True)
        Qsimave2 = Qsimmu2.mean(-1, keepdim=True)
        ETave = ETmu.mean(-1, keepdim=True)

        # get the initial component average
        if muwts is None:
            Qsimave = Qsimmu.mean(-1)
        else:
            Qsimave = (Qsimmu * muwts).sum(-1)

        if routOpt is True: # routing
            if comprout is True:
                # do routing to all the components, reshape the matrix to [Time, gage*multi]
                Qsim = Qsimmu.view(Nstep, Ngrid * mu)
            else:
                # average the components first, then do routing
                Qsim = Qsimave
            # scale learned routing parameter
            tempa = routscaLst[0][0] + rtwts[:,0]*(routscaLst[0][1]-routscaLst[0][0])
            tempb = routscaLst[1][0] + rtwts[:,1]*(routscaLst[1][1]-routscaLst[1][0])
            routa = tempa.repeat(Nstep, 1).unsqueeze(-1)
            routb = tempb.repeat(Nstep, 1).unsqueeze(-1)
            UH = UH_gamma(routa, routb, lenF=15)  # lenF: folter
            rf = torch.unsqueeze(Qsim, -1).permute([1, 2, 0])   # dim:gage*var*time
            UH = UH.permute([1, 2, 0])  # dim: gage*var*time
            Qsrout = UH_conv(rf, UH).permute([2, 0, 1])

            if comprout is True: # Qs is [time, [gage*mult], var] now
                Qstemp = Qsrout.view(Nstep, Ngrid, mu)
                if muwts is None:
                    Qs = Qstemp.mean(-1, keepdim=True)
                else:
                    Qs = (Qstemp * muwts).sum(-1, keepdim=True)
            else:
                Qs = Qsrout

        else: # no routing, output the initial average simulations

            Qs = torch.unsqueeze(Qsimave, -1) # add a dimension

        if outstate is True:
            return Qs, SNOWPACK, MELTWATER, SM, SUZ, SLZ
        else:
            Qall = torch.cat((Qs, Qsimave0, Qsimave1, Qsimave2, ETave), dim=-1)
            return Qall


class HBVMulET(torch.nn.Module):
    """Multi-component HBV Model PyTorch version"""
    # Add an ET shape parameter; others are the same as class HBVMul()
    # refer HBVMul() for detailed comments

    def __init__(self):
        """Initiate a HBV instance"""
        super(HBVMulET, self).__init__()

    def forward(self, x, parameters, mu, muwts, rtwts, bufftime=0, outstate=False, routOpt=False, comprout=False):

        PRECS = 1e-5

        # Initialization
        if bufftime > 0:
            with torch.no_grad():
                xinit = x[0:bufftime, :, :]
                initmodel = HBVMulET()
                Qsinit, SNOWPACK, MELTWATER, SM, SUZ, SLZ = initmodel(xinit, parameters, mu, muwts, rtwts,
                                                                      bufftime=0, outstate=True, routOpt=False, comprout=False)
        else:

            # Without buff time, initialize state variables with zeros
            Ngrid = x.shape[1]
            SNOWPACK = (torch.zeros([Ngrid,mu], dtype=torch.float32) + 0.001).cuda()
            MELTWATER = (torch.zeros([Ngrid,mu], dtype=torch.float32) + 0.001).cuda()
            SM = (torch.zeros([Ngrid,mu], dtype=torch.float32) + 0.001).cuda()
            SUZ = (torch.zeros([Ngrid,mu], dtype=torch.float32) + 0.001).cuda()
            SLZ = (torch.zeros([Ngrid,mu], dtype=torch.float32) + 0.001).cuda()
            # ETact = (torch.zeros([Ngrid,mu], dtype=torch.float32) + 0.001).cuda()

        P = x[bufftime:, :, 0]
        Pm= P.unsqueeze(2).repeat(1,1,mu)
        T = x[bufftime:, :, 1]
        Tm = T.unsqueeze(2).repeat(1,1,mu)
        ETpot = x[bufftime:, :, 2]
        ETpm = ETpot.unsqueeze(2).repeat(1,1,mu)


        ## scale the parameters
        parascaLst = [[1,6], [50,1000], [0.05,0.9], [0.01,0.5], [0.001,0.2], [0.2,1],
                        [0,10], [0,100], [-2.5,2.5], [0.5,10], [0,0.1], [0,0.2], [0.3,5]]
        routscaLst = [[0,2.9], [0,6.5]]

        parBETA = parascaLst[0][0] + parameters[:,0,:]*(parascaLst[0][1]-parascaLst[0][0])
        parFC = parascaLst[1][0] + parameters[:,1,:]*(parascaLst[1][1]-parascaLst[1][0])
        parK0 = parascaLst[2][0] + parameters[:,2,:]*(parascaLst[2][1]-parascaLst[2][0])
        parK1 = parascaLst[3][0] + parameters[:,3,:]*(parascaLst[3][1]-parascaLst[3][0])
        parK2 = parascaLst[4][0] + parameters[:,4,:]*(parascaLst[4][1]-parascaLst[4][0])
        parLP = parascaLst[5][0] + parameters[:,5,:]*(parascaLst[5][1]-parascaLst[5][0])
        parPERC = parascaLst[6][0] + parameters[:,6,:]*(parascaLst[6][1]-parascaLst[6][0])
        parUZL = parascaLst[7][0] + parameters[:,7,:]*(parascaLst[7][1]-parascaLst[7][0])
        parTT = parascaLst[8][0] + parameters[:,8,:]*(parascaLst[8][1]-parascaLst[8][0])
        parCFMAX = parascaLst[9][0] + parameters[:,9,:]*(parascaLst[9][1]-parascaLst[9][0])
        parCFR = parascaLst[10][0] + parameters[:,10,:]*(parascaLst[10][1]-parascaLst[10][0])
        parCWH = parascaLst[11][0] + parameters[:,11,:]*(parascaLst[11][1]-parascaLst[11][0])

        # The added ET parameter
        parBETAET = parascaLst[12][0] + parameters[:,12,:]*(parascaLst[12][1]-parascaLst[12][0])

        Nstep, Ngrid = P.size()

        # Initialize time series of model variables
        Qsimmu = (torch.zeros(Pm.size(), dtype=torch.float32) + 0.001).cuda()

        for t in range(Nstep):
            # Separate precipitation into liquid and solid components
            PRECIP = Pm[t, :, :]  # need to check later, seems repeating with line 52
            RAIN = torch.mul(PRECIP, (Tm[t, :, :] >= parTT).type(torch.float32))
            SNOW = torch.mul(PRECIP, (Tm[t, :, :] < parTT).type(torch.float32))

            # Snow
            SNOWPACK = SNOWPACK + SNOW
            melt = parCFMAX * (Tm[t, :, :] - parTT)
            # melt[melt < 0.0] = 0.0
            melt = torch.clamp(melt, min=0.0)
            # melt[melt > SNOWPACK] = SNOWPACK[melt > SNOWPACK]
            melt = torch.min(melt, SNOWPACK)
            MELTWATER = MELTWATER + melt
            SNOWPACK = SNOWPACK - melt
            refreezing = parCFR * parCFMAX * (parTT - Tm[t, :, :])
            # refreezing[refreezing < 0.0] = 0.0
            # refreezing[refreezing > MELTWATER] = MELTWATER[refreezing > MELTWATER]
            refreezing = torch.clamp(refreezing, min=0.0)
            refreezing = torch.min(refreezing, MELTWATER)
            SNOWPACK = SNOWPACK + refreezing
            MELTWATER = MELTWATER - refreezing
            tosoil = MELTWATER - (parCWH * SNOWPACK)
            # tosoil[tosoil < 0.0] = 0.0
            tosoil = torch.clamp(tosoil, min=0.0)
            MELTWATER = MELTWATER - tosoil

            # Soil and evaporation
            soil_wetness = (SM / parFC) ** parBETA
            # soil_wetness[soil_wetness < 0.0] = 0.0
            # soil_wetness[soil_wetness > 1.0] = 1.0
            soil_wetness = torch.clamp(soil_wetness, min=0.0, max=1.0)
            recharge = (RAIN + tosoil) * soil_wetness


            SM = SM + RAIN + tosoil - recharge
            excess = SM - parFC
            # excess[excess < 0.0] = 0.0
            excess = torch.clamp(excess, min=0.0)
            SM = SM - excess
            # MODIFY HERE. Add the shape para parBETAET for ET equation
            evapfactor = (SM / (parLP * parFC)) ** parBETAET
            # evapfactor = SM / (parLP * parFC)
            # evapfactor[evapfactor < 0.0] = 0.0
            # evapfactor[evapfactor > 1.0] = 1.0
            evapfactor  = torch.clamp(evapfactor, min=0.0, max=1.0)
            ETact = ETpm[t, :, :] * evapfactor
            ETact = torch.min(SM, ETact)
            SM = torch.clamp(SM - ETact, min=PRECS) # SM can not be zero for gradient tracking

            # Groundwater boxes
            SUZ = SUZ + recharge + excess
            PERC = torch.min(SUZ, parPERC)
            SUZ = SUZ - PERC
            Q0 = parK0 * torch.clamp(SUZ - parUZL, min=0.0)
            SUZ = SUZ - Q0
            Q1 = parK1 * SUZ
            SUZ = SUZ - Q1
            SLZ = SLZ + PERC
            Q2 = parK2 * SLZ
            SLZ = SLZ - Q2
            Qsimmu[t, :, :] = Q0 + Q1 + Q2


        # get the initial average
        if muwts is None:
            Qsimave = Qsimmu.mean(-1)
        else:
            Qsimave = (Qsimmu * muwts).sum(-1)

        if routOpt is True: # routing
            if comprout is True:
                # do routing to all the components, reshape the mat to [Time, gage*multi]
                Qsim = Qsimmu.view(Nstep, Ngrid * mu)
            else:
                # average the components, then do routing
                Qsim = Qsimave

            tempa = routscaLst[0][0] + rtwts[:,0]*(routscaLst[0][1]-routscaLst[0][0])
            tempb = routscaLst[1][0] + rtwts[:,1]*(routscaLst[1][1]-routscaLst[1][0])
            routa = tempa.repeat(Nstep, 1).unsqueeze(-1)
            routb = tempb.repeat(Nstep, 1).unsqueeze(-1)
            UH = UH_gamma(routa, routb, lenF=15)  # lenF: folter
            rf = torch.unsqueeze(Qsim, -1).permute([1, 2, 0])   # dim:gage*var*time
            UH = UH.permute([1, 2, 0])  # dim: gage*var*time
            Qsrout = UH_conv(rf, UH).permute([2, 0, 1])

            if comprout is True: # Qs is [time, [gage*mult], var] now
                Qstemp = Qsrout.view(Nstep, Ngrid, mu)
                if muwts is None:
                    Qs = Qstemp.mean(-1, keepdim=True)
                else:
                    Qs = (Qstemp * muwts).sum(-1, keepdim=True)
            else:
                Qs = Qsrout

        else: # no routing, output the initial average simulations

            Qs = torch.unsqueeze(Qsimave, -1) # add a dimension

        if outstate is True:
            return Qs, SNOWPACK, MELTWATER, SM, SUZ, SLZ
        else:
            return Qs # total streamflow


class MultiInv_HBVModel(torch.nn.Module):
    # class for dPL + HBV with multiple components and static parameters
    def __init__(self, *, ninv, nfea, nmul, hiddeninv, drinv=0.5, inittime=0, routOpt=False, comprout=False,
                 compwts=False, pcorr=None):
        # LSTM Inv + HBV Forward
        super(MultiInv_HBVModel, self).__init__()
        self.ninv = ninv
        self.nfea = nfea
        self.hiddeninv = hiddeninv
        self.nmul = nmul
        # get the total number of parameters
        nhbvpm = nfea*nmul
        if comprout is False:
            nroutpm = 2
        else:
            nroutpm = nmul*2
        if compwts is False:
            nwtspm = 0
        else:
            nwtspm = nmul
        if pcorr is None:
            ntp = nhbvpm + nroutpm + nwtspm
        else:
            ntp = nhbvpm + nroutpm + nwtspm + 1 # 1 for potential precipitation correction
        # ntp = nfea*nmul+nmul+2
        # ntp = nfea * nmul + 2
        self.lstminv = CudnnLstmModel(
            nx=ninv, ny=ntp, hiddenSize=hiddeninv, dr=drinv)

        self.HBV = HBVMul()

        self.gpu = 1
        self.inittime=inittime
        self.routOpt=routOpt
        self.comprout=comprout
        self.nhbvpm = nhbvpm
        self.nwtspm = nwtspm
        self.nroutpm = nroutpm
        self.pcorr = pcorr



    def forward(self, x, z, doDropMC=False):
        Gen = self.lstminv(z)
        Params0 = Gen[-1, :, :] # the last time step as learned parameters
        ngage = Params0.shape[0]
        # print(Params0)
        hbvpara0 = Params0[:, 0:self.nhbvpm]
        hbvpara = torch.sigmoid(hbvpara0).view(ngage, self.nfea, self.nmul)
        routpara0 = Params0[:, self.nhbvpm:self.nhbvpm+self.nroutpm] # dim: [Ngage, nmul*2] or [Ngage, 2]
        if self.comprout is False: # if do routing for each component
            routpara = torch.sigmoid(routpara0)
        else:
            routpara = torch.sigmoid(routpara0).view(ngage * self.nmul, 2)
        if self.nwtspm == 0: # 0: simple average instead of weighted average for all components
            wts = None
        else:
            wtspara = Params0[:, self.nhbvpm+self.nroutpm:self.nhbvpm+self.nroutpm+self.nwtspm]
            wts = F.softmax(wtspara, dim=-1)
        if self.pcorr is None:
            corrpara = None
        else:
            corrpara0 = Params0[:, self.nhbvpm+self.nroutpm+self.nwtspm:self.nhbvpm+self.nroutpm+self.nwtspm+1]
            corrpara = torch.sigmoid(corrpara0)
        out = self.HBV(x, parameters=hbvpara, mu=self.nmul, muwts=wts, rtwts=routpara, bufftime=self.inittime,
                       routOpt=self.routOpt, comprout=self.comprout, corrwts=corrpara, pcorr=self.pcorr) # HBV forward
        return out


class HBVMulTD(torch.nn.Module):
    """HBV Model with multiple components and dynamic parameters PyTorch version implemented by Dapeng Feng"""
    # we suggest you read the class HBVMul() with static parameters first

    def __init__(self):
        """Initiate a HBV instance"""
        super(HBVMulTD, self).__init__()

    def forward(self, x, parameters, staind, tdlst, mu, muwts, rtwts, bufftime=0, outstate=False, routOpt=False,
                comprout=False, dydrop=False):
        # Modified from the original numpy version from Beck et al., 2020. (http://www.gloh2o.org/hbv/) which
        # runs the HBV-light hydrological model (Seibert, 2005).
        # NaN values have to be removed from the inputs.
        #
        # Input:
        #     x: dim=[time, basin, var] forcing array with var P(mm/d), T(deg C), PET(mm/d)
        #     parameters: array with parameter values having the following structure and scales:
        #         BETA[1,6]; FC[50,1000]; K0[0.05,0.9]; K1[0.01,0.5]; K2[0.001,0.2]; LP[0.2,1];
        #         PERC[0,10]; UZL[0,100]; TT[-2.5,2.5]; CFMAX[0.5,10]; CFR[0,0.1]; CWH[0,0.2]
        #     mu:number of components; muwts: weights of components if True; rtwts: routing parameters;
        #     bufftime:warm up period; outstate: output state var; routOpt:routing option; comprout:component routing opt
        #     dydrop: the possibility to drop a dynamic para to static to reduce potential overfitting
        #
        #
        # Output, all in mm:
        #     outstate True: output most state variables for warm-up
        #      Qs:simulated streamflow; SNOWPACK:snow depth; MELTWATER:snow water holding depth;
        #      SM:soil storage; SUZ:upper zone storage; SLZ:lower zone storage
        #     outstate False: output the simulated flux array Qall contains
        #      Qs:simulated streamflow=Q0+Q1+Q2; Qsimave0:Q0 component; Qsimave1:Q1 component; Qsimave2:Q2 baseflow componnet
        #      ETave: actual ET


        PRECS = 1e-5

        # Initialization
        if bufftime > 0:
            with torch.no_grad():
                xinit = x[0:bufftime, :, :]
                initmodel = HBVMul()
                buffpara = parameters[bufftime-1, :, :, :]
                Qsinit, SNOWPACK, MELTWATER, SM, SUZ, SLZ = initmodel(xinit, buffpara, mu, muwts, rtwts,
                                                                      bufftime=0, outstate=True, routOpt=False, comprout=False)
        else:

            # Without buff time, initialize state variables with zeros
            Ngrid = x.shape[1]
            SNOWPACK = (torch.zeros([Ngrid,mu], dtype=torch.float32) + 0.001).cuda()
            MELTWATER = (torch.zeros([Ngrid,mu], dtype=torch.float32) + 0.001).cuda()
            SM = (torch.zeros([Ngrid,mu], dtype=torch.float32) + 0.001).cuda()
            SUZ = (torch.zeros([Ngrid,mu], dtype=torch.float32) + 0.001).cuda()
            SLZ = (torch.zeros([Ngrid,mu], dtype=torch.float32) + 0.001).cuda()
            # ETact = (torch.zeros([Ngrid,mu], dtype=torch.float32) + 0.001).cuda()

        P = x[bufftime:, :, 0]
        Pm= P.unsqueeze(2).repeat(1,1,mu)
        T = x[bufftime:, :, 1]
        Tm = T.unsqueeze(2).repeat(1,1,mu)
        ETpot = x[bufftime:, :, 2]
        ETpm = ETpot.unsqueeze(2).repeat(1,1,mu)
        parAll = parameters[bufftime:, :, :, :]
        parAllTrans = torch.zeros_like(parAll)

        ## scale the parameters
        hbvscaLst = [[1,6], [50,1000], [0.05,0.9], [0.01,0.5], [0.001,0.2], [0.2,1],
                        [0,10], [0,100], [-2.5,2.5], [0.5,10], [0,0.1], [0,0.2]]
        routscaLst = [[0,2.9], [0,6.5]]

        for ip in range(len(hbvscaLst)): # not include routing. Scaling the parameters using loop
            parAllTrans[:,:,ip,:] = hbvscaLst[ip][0] + parAll[:,:,ip,:]*(hbvscaLst[ip][1]-hbvscaLst[ip][0])

        Nstep, Ngrid = P.size()

        # deal with the dynamic parameters and dropout
        parstaFull = parAllTrans[staind, :, :, :].unsqueeze(0).repeat([Nstep, 1, 1, 1])  # static matrix
        parhbvFull = torch.clone(parstaFull)
        # create probability mask for each parameter on the basin dimension to apply dropout
        pmat = torch.ones([1, Ngrid, 1])*dydrop
        for ix in tdlst:
            staPar = parstaFull[:, :, ix-1, :]
            dynPar = parAllTrans[:, :, ix-1, :]
            drmask = torch.bernoulli(pmat).detach_().cuda()  # to drop some dynamic parameters as static
            comPar = dynPar*(1-drmask) + staPar*drmask
            parhbvFull[:, :, ix-1, :] = comPar


        # Initialize time series of model variables
        Qsimmu = (torch.zeros(Pm.size(), dtype=torch.float32) + 0.001).cuda()
        ETmu = (torch.zeros(Pm.size(), dtype=torch.float32) + 0.001).cuda()

        # Output the box components of Q
        Qsimmu0 = (torch.zeros(Pm.size(), dtype=torch.float32) + 0.001).cuda()
        Qsimmu1 = (torch.zeros(Pm.size(), dtype=torch.float32) + 0.001).cuda()
        Qsimmu2 = (torch.zeros(Pm.size(), dtype=torch.float32) + 0.001).cuda()


        for t in range(Nstep):
            paraLst = []
            for ip in range(len(hbvscaLst)):  # unpack HBV parameters
                paraLst.append(parhbvFull[t, :, ip, :])

            parBETA, parFC, parK0, parK1, parK2, parLP, parPERC, parUZL, parTT, parCFMAX, parCFR, parCWH = paraLst

            # Separate precipitation into liquid and solid components
            PRECIP = Pm[t, :, :]
            RAIN = torch.mul(PRECIP, (Tm[t, :, :] >= parTT).type(torch.float32))
            SNOW = torch.mul(PRECIP, (Tm[t, :, :] < parTT).type(torch.float32))

            # Snow
            SNOWPACK = SNOWPACK + SNOW
            melt = parCFMAX * (Tm[t, :, :] - parTT)
            # melt[melt < 0.0] = 0.0
            melt = torch.clamp(melt, min=0.0)
            # melt[melt > SNOWPACK] = SNOWPACK[melt > SNOWPACK]
            melt = torch.min(melt, SNOWPACK)
            MELTWATER = MELTWATER + melt
            SNOWPACK = SNOWPACK - melt
            refreezing = parCFR * parCFMAX * (parTT - Tm[t, :, :])
            # refreezing[refreezing < 0.0] = 0.0
            # refreezing[refreezing > MELTWATER] = MELTWATER[refreezing > MELTWATER]
            refreezing = torch.clamp(refreezing, min=0.0)
            refreezing = torch.min(refreezing, MELTWATER)
            SNOWPACK = SNOWPACK + refreezing
            MELTWATER = MELTWATER - refreezing
            tosoil = MELTWATER - (parCWH * SNOWPACK)
            # tosoil[tosoil < 0.0] = 0.0
            tosoil = torch.clamp(tosoil, min=0.0)
            MELTWATER = MELTWATER - tosoil

            # Soil and evaporation
            soil_wetness = (SM / parFC) ** parBETA
            # soil_wetness[soil_wetness < 0.0] = 0.0
            # soil_wetness[soil_wetness > 1.0] = 1.0
            soil_wetness = torch.clamp(soil_wetness, min=0.0, max=1.0)
            recharge = (RAIN + tosoil) * soil_wetness

            SM = SM + RAIN + tosoil - recharge
            excess = SM - parFC
            # excess[excess < 0.0] = 0.0
            excess = torch.clamp(excess, min=0.0)
            SM = SM - excess
            evapfactor = SM / (parLP * parFC)
            evapfactor  = torch.clamp(evapfactor, min=0.0, max=1.0)
            ETact = ETpm[t, :, :] * evapfactor
            ETact = torch.min(SM, ETact)
            SM = torch.clamp(SM - ETact, min=PRECS) # SM can not be zero for gradient tracking

            # Groundwater boxes
            SUZ = SUZ + recharge + excess
            PERC = torch.min(SUZ, parPERC)
            SUZ = SUZ - PERC
            Q0 = parK0 * torch.clamp(SUZ - parUZL, min=0.0)
            SUZ = SUZ - Q0
            Q1 = parK1 * SUZ
            SUZ = SUZ - Q1
            SLZ = SLZ + PERC
            Q2 = parK2 * SLZ
            SLZ = SLZ - Q2
            Qsimmu[t, :, :] = Q0 + Q1 + Q2

            # save components
            Qsimmu0[t, :, :] = Q0
            Qsimmu1[t, :, :] = Q1
            Qsimmu2[t, :, :] = Q2
            ETmu[t, :, :] = ETact


        Qsimave0 = Qsimmu0.mean(-1, keepdim=True)
        Qsimave1 = Qsimmu1.mean(-1, keepdim=True)
        Qsimave2 = Qsimmu2.mean(-1, keepdim=True)
        ETave = ETmu.mean(-1, keepdim=True)


        # get the initial average
        if muwts is None:
            Qsimave = Qsimmu.mean(-1)
        else:
            Qsimave = (Qsimmu * muwts).sum(-1)

        if routOpt is True: # routing
            if comprout is True:
                # do routing to all the components, reshape the mat to [Time, gage*multi]
                Qsim = Qsimmu.view(Nstep, Ngrid * mu)
            else:
                # average the components, then do routing
                Qsim = Qsimave

            tempa = routscaLst[0][0] + rtwts[:,0]*(routscaLst[0][1]-routscaLst[0][0])
            tempb = routscaLst[1][0] + rtwts[:,1]*(routscaLst[1][1]-routscaLst[1][0])
            routa = tempa.repeat(Nstep, 1).unsqueeze(-1)
            routb = tempb.repeat(Nstep, 1).unsqueeze(-1)
            UH = UH_gamma(routa, routb, lenF=15)  # lenF: folter
            rf = torch.unsqueeze(Qsim, -1).permute([1, 2, 0])   # dim:gage*var*time
            UH = UH.permute([1, 2, 0])  # dim: gage*var*time
            Qsrout = UH_conv(rf, UH).permute([2, 0, 1])

            if comprout is True: # Qs is [time, [gage*mult], var] now
                Qstemp = Qsrout.view(Nstep, Ngrid, mu)
                if muwts is None:
                    Qs = Qstemp.mean(-1, keepdim=True)
                else:
                    Qs = (Qstemp * muwts).sum(-1, keepdim=True)
            else:
                Qs = Qsrout

        else: # no routing, output the primary average simulations

            Qs = torch.unsqueeze(Qsimave, -1) # add a dimension

        if outstate is True:
            return Qs, SNOWPACK, MELTWATER, SM, SUZ, SLZ
        else:
            # return Qs
            Qall = torch.cat((Qs, Qsimave0, Qsimave1, Qsimave2, ETave), dim=-1)
            return Qall

class HBVMulTDET(torch.nn.Module):
    """HBV Model with multiple components and dynamic parameters PyTorch version"""
    # Add an ET shape parameter for the original ET equation; others are the same as HBVMulTD()
    # we suggest you read the class HBVMul() with original static parameters first

    def __init__(self):
        """Initiate a HBV instance"""
        super(HBVMulTDET, self).__init__()

    def forward(self, x, parameters, staind, tdlst, mu, muwts, rtwts, bufftime=0, outstate=False, routOpt=False,
                comprout=False, dydrop=False):
        # Modified from the original numpy version from Beck et al., 2020. (http://www.gloh2o.org/hbv/) which
        # runs the HBV-light hydrological model (Seibert, 2005).
        # NaN values have to be removed from the inputs.
        #
        # Input:
        #     X: dim=[time, basin, var] forcing array with var P(mm/d), T(deg C), PET(mm/d)
        #     parameters: array with parameter values having the following structure and scales:
        #         BETA[1,6]; FC[50,1000]; K0[0.05,0.9]; K1[0.01,0.5]; K2[0.001,0.2]; LP[0.2,1];
        #         PERC[0,10]; UZL[0,100]; TT[-2.5,2.5]; CFMAX[0.5,10]; CFR[0,0.1]; CWH[0,0.2]
        #     staind:use which time step from the learned para time series for static parameters
        #     tdlst: the index list of hbv parameters set as dynamic
        #     mu:number of components; muwts: weights of components if True; rtwts: routing parameters;
        #     bufftime:warm up period; outstate: output state var; routOpt:routing option; comprout:component routing opt
        #     dydrop: the possibility to drop a dynamic para to static to reduce potential overfitting
        #
        #
        # Output, all in mm:
        #     outstate True: output most state variables for warm-up
        #      Qs:simulated streamflow; SNOWPACK:snow depth; MELTWATER:snow water holding depth;
        #      SM:soil storage; SUZ:upper zone storage; SLZ:lower zone storage
        #     outstate False: output the simulated flux array Qall contains
        #      Qs:simulated streamflow=Q0+Q1+Q2; Qsimave0:Q0 component; Qsimave1:Q1 component; Qsimave2:Q2 baseflow componnet
        #      ETave: actual ET

        PRECS = 1e-5  # keep the numerical calculation stable

        # Initialization to warm-up states
        if bufftime > 0:
            with torch.no_grad():
                xinit = x[0:bufftime, :, :]
                initmodel = HBVMulET()
                buffpara = parameters[bufftime-1, :, :, :]
                Qsinit, SNOWPACK, MELTWATER, SM, SUZ, SLZ = initmodel(xinit, buffpara, mu, muwts, rtwts,
                                                                      bufftime=0, outstate=True, routOpt=False, comprout=False)
        else:

            # Without warm-up bufftime=0, initialize state variables with zeros
            Ngrid = x.shape[1]
            SNOWPACK = (torch.zeros([Ngrid,mu], dtype=torch.float32) + 0.001).cuda()
            MELTWATER = (torch.zeros([Ngrid,mu], dtype=torch.float32) + 0.001).cuda()
            SM = (torch.zeros([Ngrid,mu], dtype=torch.float32) + 0.001).cuda()
            SUZ = (torch.zeros([Ngrid,mu], dtype=torch.float32) + 0.001).cuda()
            SLZ = (torch.zeros([Ngrid,mu], dtype=torch.float32) + 0.001).cuda()
            # ETact = (torch.zeros([Ngrid,mu], dtype=torch.float32) + 0.001).cuda()

        P = x[bufftime:, :, 0]
        Pm= P.unsqueeze(2).repeat(1,1,mu) # precip
        T = x[bufftime:, :, 1]
        Tm = T.unsqueeze(2).repeat(1,1,mu) # temperature
        ETpot = x[bufftime:, :, 2]
        ETpm = ETpot.unsqueeze(2).repeat(1,1,mu) # potential ET
        parAll = parameters[bufftime:, :, :, :]
        parAllTrans = torch.zeros_like(parAll)

        ## scale the parameters to real values from [0,1]
        hbvscaLst = [[1,6], [50,1000], [0.05,0.9], [0.01,0.5], [0.001,0.2], [0.2,1],
                        [0,10], [0,100], [-2.5,2.5], [0.5,10], [0,0.1], [0,0.2], [0.3,5]]  # HBV para
        routscaLst = [[0,2.9], [0,6.5]]  # routing para

        for ip in range(len(hbvscaLst)): # not include routing. Scaling the parameters
            parAllTrans[:,:,ip,:] = hbvscaLst[ip][0] + parAll[:,:,ip,:]*(hbvscaLst[ip][1]-hbvscaLst[ip][0])

        Nstep, Ngrid = P.size()

        # deal with the dynamic parameters and dropout to reduce overfitting of dynamic para
        parstaFull = parAllTrans[staind, :, :, :].unsqueeze(0).repeat([Nstep, 1, 1, 1])  # static para matrix
        parhbvFull = torch.clone(parstaFull)
        # create probability mask for each parameter on the basin dimension
        pmat = torch.ones([1, Ngrid, 1])*dydrop
        for ix in tdlst:
            staPar = parstaFull[:, :, ix-1, :]
            dynPar = parAllTrans[:, :, ix-1, :]
            drmask = torch.bernoulli(pmat).detach_().cuda()  # to drop dynamic parameters as static in some basins
            comPar = dynPar*(1-drmask) + staPar*drmask
            parhbvFull[:, :, ix-1, :] = comPar


        # Initialize time series of model variables to save results
        Qsimmu = (torch.zeros(Pm.size(), dtype=torch.float32) + 0.001).cuda()
        ETmu = (torch.zeros(Pm.size(), dtype=torch.float32) + 0.001).cuda()

        # Output the box components of Q
        Qsimmu0 = (torch.zeros(Pm.size(), dtype=torch.float32) + 0.001).cuda()
        Qsimmu1 = (torch.zeros(Pm.size(), dtype=torch.float32) + 0.001).cuda()
        Qsimmu2 = (torch.zeros(Pm.size(), dtype=torch.float32) + 0.001).cuda()

        # # Not used. Logging the state variables for debug.
        # # SMlog = np.zeros(P.size())
        # logSM = np.zeros(P.size())
        # logPS = np.zeros(P.size())
        # logswet = np.zeros(P.size())
        # logRE = np.zeros(P.size())

        for t in range(Nstep):
            paraLst = []
            for ip in range(len(hbvscaLst)):  # unpack HBV parameters
                paraLst.append(parhbvFull[t, :, ip, :])

            parBETA, parFC, parK0, parK1, parK2, parLP, parPERC, parUZL, parTT, parCFMAX, parCFR, parCWH, parBETAET = paraLst
            # Separate precipitation into liquid and solid components
            PRECIP = Pm[t, :, :]
            RAIN = torch.mul(PRECIP, (Tm[t, :, :] >= parTT).type(torch.float32))
            SNOW = torch.mul(PRECIP, (Tm[t, :, :] < parTT).type(torch.float32))

            # Snow process
            SNOWPACK = SNOWPACK + SNOW
            melt = parCFMAX * (Tm[t, :, :] - parTT)
            melt = torch.clamp(melt, min=0.0)
            melt = torch.min(melt, SNOWPACK)
            MELTWATER = MELTWATER + melt
            SNOWPACK = SNOWPACK - melt
            refreezing = parCFR * parCFMAX * (parTT - Tm[t, :, :])
            refreezing = torch.clamp(refreezing, min=0.0)
            refreezing = torch.min(refreezing, MELTWATER)
            SNOWPACK = SNOWPACK + refreezing
            MELTWATER = MELTWATER - refreezing
            tosoil = MELTWATER - (parCWH * SNOWPACK)
            tosoil = torch.clamp(tosoil, min=0.0)
            MELTWATER = MELTWATER - tosoil

            # Soil and evaporation
            soil_wetness = (SM / parFC) ** parBETA
            soil_wetness = torch.clamp(soil_wetness, min=0.0, max=1.0)
            recharge = (RAIN + tosoil) * soil_wetness

            # Not used, logging states for checking
            # logSM[t,:] = SM.detach().cpu().numpy()
            # logPS[t,:] = (RAIN + tosoil).detach().cpu().numpy()
            # logswet[t,:] = (SM / parFC).detach().cpu().numpy()
            # logRE[t, :] = recharge.detach().cpu().numpy()

            SM = SM + RAIN + tosoil - recharge
            excess = SM - parFC
            excess = torch.clamp(excess, min=0.0)
            SM = SM - excess
            # MODIFY here. Different as class HBVMulT(). Add a ET shape parameter parBETAET
            evapfactor = (SM / (parLP * parFC)) ** parBETAET
            evapfactor  = torch.clamp(evapfactor, min=0.0, max=1.0)
            ETact = ETpm[t, :, :] * evapfactor
            ETact = torch.min(SM, ETact)
            SM = torch.clamp(SM - ETact, min=PRECS) # SM can not be zero for gradient tracking

            # Groundwater boxes
            SUZ = SUZ + recharge + excess
            PERC = torch.min(SUZ, parPERC)
            SUZ = SUZ - PERC
            Q0 = parK0 * torch.clamp(SUZ - parUZL, min=0.0)
            SUZ = SUZ - Q0
            Q1 = parK1 * SUZ
            SUZ = SUZ - Q1
            SLZ = SLZ + PERC
            Q2 = parK2 * SLZ
            SLZ = SLZ - Q2
            Qsimmu[t, :, :] = Q0 + Q1 + Q2

            # save components for Q
            Qsimmu0[t, :, :] = Q0
            Qsimmu1[t, :, :] = Q1
            Qsimmu2[t, :, :] = Q2
            ETmu[t, :, :] = ETact

            # Not used, for debug state variables
            # SMlog[t,:, :] = SM.detach().cpu().numpy()
            # SUZlog[t,:,:] = SUZ.detach().cpu().numpy()
            # SLZlog[t,:,:] = SLZ.detach().cpu().numpy()

        Qsimave0 = Qsimmu0.mean(-1, keepdim=True)
        Qsimave1 = Qsimmu1.mean(-1, keepdim=True)
        Qsimave2 = Qsimmu2.mean(-1, keepdim=True)
        ETave = ETmu.mean(-1, keepdim=True)

        # get the initial average
        if muwts is None: # simple average
            Qsimave = Qsimmu.mean(-1)
        else: # weighted average using learned weights
            Qsimave = (Qsimmu * muwts).sum(-1)

        if routOpt is True: # routing
            if comprout is True:
                # do routing to all the components, reshape the mat to [Time, gage*multi]
                Qsim = Qsimmu.view(Nstep, Ngrid * mu)
            else:
                # average the components, then do routing
                Qsim = Qsimave

            # scale two routing parameters
            tempa = routscaLst[0][0] + rtwts[:,0]*(routscaLst[0][1]-routscaLst[0][0])
            tempb = routscaLst[1][0] + rtwts[:,1]*(routscaLst[1][1]-routscaLst[1][0])
            routa = tempa.repeat(Nstep, 1).unsqueeze(-1)
            routb = tempb.repeat(Nstep, 1).unsqueeze(-1)
            UH = UH_gamma(routa, routb, lenF=15)  # lenF: folter
            rf = torch.unsqueeze(Qsim, -1).permute([1, 2, 0])   # dim:gage*var*time
            UH = UH.permute([1, 2, 0])  # dim: gage*var*time
            Qsrout = UH_conv(rf, UH).permute([2, 0, 1])

            if comprout is True: # Qs is [time, [gage*mult], var] now
                Qstemp = Qsrout.view(Nstep, Ngrid, mu)
                if muwts is None:
                    Qs = Qstemp.mean(-1, keepdim=True)
                else:
                    Qs = (Qstemp * muwts).sum(-1, keepdim=True)
            else:
                Qs = Qsrout

        else: # no routing, output the initial average simulations
            Qs = torch.unsqueeze(Qsimave, -1) # add a dimension

        if outstate is True: # output states
            return Qs, SNOWPACK, MELTWATER, SM, SUZ, SLZ
        else:
            # return Qs
            Qall = torch.cat((Qs, Qsimave0, Qsimave1, Qsimave2, ETave), dim=-1)
            return Qall



class MultiInv_HBVTDModel(torch.nn.Module):
    # class for dPL + HBV with multiple components and some dynamic parameters
    def __init__(self, *, ninv, nfea, nmul, hiddeninv, drinv=0.5, inittime=0, routOpt=False, comprout=False,
                 compwts=False, staind=-1, tdlst=[], dydrop=0.0, ETMod=False):
        # LSTM Inv + HBV Forward
        super(MultiInv_HBVTDModel, self).__init__()
        self.ninv = ninv
        self.nfea = nfea
        self.hiddeninv = hiddeninv
        self.nmul = nmul
        # get the total number of parameters
        nhbvpm = nfea*nmul
        if comprout is False:
            nroutpm = 2
        else:
            nroutpm = nmul*2
        if compwts is False:
            nwtspm = 0
        else:
            nwtspm = nmul
        ntp = nhbvpm + nroutpm + nwtspm
        # ntp = nfea*nmul+nmul+2
        # ntp = nfea * nmul + 2
        self.lstminv = CudnnLstmModel(
            nx=ninv, ny=ntp, hiddenSize=hiddeninv, dr=drinv)

        if ETMod is True:
            # use the added para for ET eq
            self.HBV = HBVMulTDET()
        else:
            # the original HBV para
            self.HBV = HBVMulTD()

        self.gpu = 1
        self.inittime=inittime
        self.routOpt=routOpt
        self.comprout=comprout
        self.nhbvpm = nhbvpm
        self.nwtspm = nwtspm
        self.nroutpm = nroutpm
        self.staind = staind
        self.tdlst = tdlst
        self.dydrop = dydrop


    def forward(self, x, z, doDropMC=False):
        Params0 = self.lstminv(z) # dim: Time, Gage, Para
        ntstep = Params0.shape[0]
        ngage = Params0.shape[1]
        # print(Params0)
        hbvpara0 = Params0[:, :, 0:self.nhbvpm]
        # hbvpara = torch.clamp(hbvpara0, min=0.0, max=1.0).view(ngage, self.nfea, self.nmul)
        hbvpara = torch.sigmoid(hbvpara0).view(ntstep, ngage, self.nfea, self.nmul) # hbv scaled para, [0,1]
        routpara0 = Params0[-1, :, self.nhbvpm:self.nhbvpm+self.nroutpm] # routing para dim:[Ngage, nmul*2] or [Ngage, 2]
        if self.comprout is False:
            # routpara = torch.clamp(routpara0, min=0.0, max=1.0)
            routpara = torch.sigmoid(routpara0) # [0,1]
        else:
            # routpara = torch.clamp(routpara0, min=0.0, max=1.0).view(ngage*self.nmul, 2) # first dim:gage*component
            routpara = torch.sigmoid(routpara0).view(ngage * self.nmul, 2) # [0,1]
        if self.nwtspm == 0: # simple average for multiple components
            wts = None
        else: # weighted average using learned weights
            wtspara = Params0[-1, :, -self.nwtspm:]
            wts = F.softmax(wtspara, dim=-1)  # softmax to make sure sum to 1
        out = self.HBV(x, parameters=hbvpara, staind=self.staind, tdlst=self.tdlst, mu=self.nmul, muwts=wts, rtwts=routpara,
                          bufftime=self.inittime, routOpt=self.routOpt, comprout=self.comprout, dydrop=self.dydrop)
        return out



class dHBVModel(torch.nn.Module):
    # class for dPL + HBV with multiple components and some dynamic parameters for both dHBV1.0 and dHBV1.1p
    def __init__(self, *, ninv, nfea, nmul, hiddeninv, drinv=0.5, inittime=0, routOpt=False, comprout=False,
                 compwts=False, staind=-1, tdlst=[], dydrop=0.0, ETMod=False, model_name = 'HBV1_0'):
        # LSTM Inv + HBV Forward
        super(dHBVModel, self).__init__()
        package_name = "hydroDL.model.HydroModels"
        model_import_string = f"{package_name}.{model_name}"

        try:
            PBMmodel = getattr(importlib.import_module(model_import_string), 'HBV')
        except ImportError:
            print(f"Failed to import {model_import_string} from {model_name}")


        self.ninv = ninv
        self.nfea = nfea
        self.hiddeninv = hiddeninv
        self.nmul = nmul
        # get the total number of parameters
        nhbvpm = nfea*nmul
        if comprout is False:
            nroutpm = 2
        else:
            nroutpm = nmul*2
        if compwts is False:
            nwtspm = 0
        else:
            nwtspm = nmul
        ntp = nhbvpm + nroutpm + nwtspm
        # ntp = nfea*nmul+nmul+2
        # ntp = nfea * nmul + 2
        self.lstminv = CudnnLstmModel(
            nx=ninv, ny=ntp, hiddenSize=hiddeninv, dr=drinv)


        self.HBV = PBMmodel()



        self.inittime=inittime
        self.routOpt=routOpt
        self.comprout=comprout
        self.nhbvpm = nhbvpm
        self.nwtspm = nwtspm
        self.nroutpm = nroutpm
        self.staind = staind
        self.tdlst = tdlst
        self.dydrop = dydrop


    def forward(self, x, z, doDropMC=False):
        Params0 = self.lstminv(z) # dim: Time, Gage, Para
        ntstep = Params0.shape[0]
        ngage = Params0.shape[1]
        # print(Params0)
        hbvpara0 = Params0[:, :, 0:self.nhbvpm]
        # hbvpara = torch.clamp(hbvpara0, min=0.0, max=1.0).view(ngage, self.nfea, self.nmul)
        hbvpara = torch.sigmoid(hbvpara0).view(ntstep, ngage, self.nfea, self.nmul) # hbv scaled para, [0,1]
        routpara0 = Params0[-1, :, self.nhbvpm:self.nhbvpm+self.nroutpm] # routing para dim:[Ngage, nmul*2] or [Ngage, 2]
        if self.comprout is False:
            # routpara = torch.clamp(routpara0, min=0.0, max=1.0)
            routpara = torch.sigmoid(routpara0) # [0,1]
        else:
            # routpara = torch.clamp(routpara0, min=0.0, max=1.0).view(ngage*self.nmul, 2) # first dim:gage*component
            routpara = torch.sigmoid(routpara0).view(ngage * self.nmul, 2) # [0,1]
        if self.nwtspm == 0: # simple average for multiple components
            wts = None
        else: # weighted average using learned weights
            wtspara = Params0[-1, :, -self.nwtspm:]
            wts = F.softmax(wtspara, dim=-1)  # softmax to make sure sum to 1
        out = self.HBV(x, parameters=hbvpara, staind=self.staind, tdlst=self.tdlst, mu=self.nmul, muwts=wts, rtwts=routpara,
                          bufftime=self.inittime, routOpt=self.routOpt, comprout=self.comprout, dydrop=self.dydrop)
        return out

