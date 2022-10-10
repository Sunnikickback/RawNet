import importlib

from torch import nn


class SpeakerNet(nn.Module):
    def __init__(self, model,  nPerSpeaker=1, **kwargs):
        super(SpeakerNet, self).__init__()

        SpeakerNetModel = importlib.import_module("models." + model).__getattribute__("MainModel")

        self.__S__ = SpeakerNetModel(nOut=512)

        self.nPerSpeaker = nPerSpeaker

    def forward(self, data, label=None):

        data = data.reshape(-1, data.size()[-1]).cuda()
        outp = self.__S__.forward(data)

        if label == None:
            return outp

        else:

            outp = outp.reshape(self.nPerSpeaker, -1, outp.size()[-1]).transpose(1, 0).squeeze(1)

            nloss, prec1 = self.__L__.forward(outp, label)

            return nloss, prec1