import unittest
import torch
import librosa
import soundfile

from torch.nn import BCELoss as Loss
from torch.optim import SGD as Opt

from lr_speech import SpeechDataset, create_dataset, step, SimpleLogreg
from numpy import array

# specify a small number of files
soundfile_dict = {'ae': ['./Hillenbrand/women/w01ae.wav','./Hillenbrand/men/m21ae.wav'],
                  'eh': ['./Hillenbrand/women/w42eh.wav','./Hillenbrand/men/m02eh.wav']}
vowels = ['ae','eh']

class TestPyTorchLR(unittest.TestCase):
    def setUp(self):
        # need to change to be MFCC values of these files
        self.raw_data = array([[-1.18445675, -1.08867637, 0.63397533, -1.56891473,
                                -0.38190164, 0.69477824, 1.12961515, 1.70717832,
                                -1.57606494, -0.34355823, -0.7872916, -1.16346313,
                                -0.21992805],
                               [0.0246291, 0.12129877, -1.59870044, 0.98698212,
                                0.33368, -0.85702608, -1.55949485, -0.50839241,
                                -0.1518257, 1.61104793, -0.65618443, 1.59747967,
                                -1.27045424],
                               [-0.40086534, -0.5961632, 1.01943095, -0.1534716,
                                -1.34366952, 1.25790498, -0.10092969, -0.36643855,
                                0.87303459, -1.12528693, -0.25493553, -0.20635286,
                                -0.03805785],
                               [1.56069299, 1.56354079, -0.05470585, 0.7354042,
                                1.39189116, -1.09565714, 0.5308094, -0.83234736,
                                0.85485605, -0.14220277,  1.69841155, -0.22766368,
                                1.52844014]])
        # not sure if this needs to be changed at all
        self.data = torch.from_numpy(self.raw_data).float()

        labels = array([[0], [0], [1], [1]])
        self.labels = torch.from_numpy(labels).float()
        
        self.model = SimpleLogreg(13)

        with torch.no_grad():
            self.model.linear.weight.fill_(0)
            self.model.linear.weight[0,0] = 1
            self.model.linear.weight[0,4] = -1        

    def test_data(self):
        s = create_dataset(soundfile_dict,vowels,13)
        for ii in range(2):
            for jj in range(13):
                self.assertAlmostEqual(s[ii][jj+1], self.raw_data[ii][jj])

    def test_forward(self):
        self.assertAlmostEqual(0.30474069714546204, float(self.model.forward(self.data)[0]))
        self.assertAlmostEqual(0.4179196059703827, float(self.model.forward(self.data)[1]))
        self.assertAlmostEqual(0.7151512503623962, float(self.model.forward(self.data)[2]))
        self.assertAlmostEqual(0.5365679264068604, float(self.model.forward(self.data)[3]))

    def test_step(self):
        optimizer = Opt(self.model.parameters(), lr=0.1)
        criterion = Loss()
        step(0, 0, self.model, optimizer, criterion, self.data, self.labels)

        weight, bias = list(self.model.parameters())
        
        self.assertAlmostEqual(float(weight[0][0]), 1.0239, 3)
        self.assertAlmostEqual(float(weight[0][1]), 0.0207, 3)
        self.assertAlmostEqual(float(weight[0][2]), 0.0185, 3)
        self.assertAlmostEqual(float(weight[0][3]), 0.0089, 3)
        self.assertAlmostEqual(float(weight[0][4]), -0.9941, 3)
        self.assertAlmostEqual(float(weight[0][5]), 0.0000, 3)
        self.assertAlmostEqual(float(weight[0][6]), 0.0131, 3)
        self.assertAlmostEqual(float(weight[0][7]), -0.0198, 3)
        self.assertAlmostEqual(float(weight[0][8]), 0.0296, 3)
        self.assertAlmostEqual(float(weight[0][9]), -0.0239, 3)
        self.assertAlmostEqual(float(weight[0][10]), 0.0306, 3)
        self.assertAlmostEqual(float(weight[0][11]), -0.0120, 3)
        self.assertAlmostEqual(float(weight[0][12]), 0.0323, 3)

        self.assertAlmostEqual(float(bias[0]), 0.03975, 3)
        
if __name__ == '__main__':
    unittest.main()
