import os
import gzip
import numpy as np
import urllib.request

class DatasetLoader:
    """
    Unified loader with high-reliability mirrors.
    """
    URLS = {
        'mnist': {
            'train_img': 'https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz',
            'train_lbl': 'https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz',
            'test_img': 'https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz',
            'test_lbl': 'https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz'
        },
        'fashion': {
            'train_img': 'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz',
            'train_lbl': 'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz',
            'test_img': 'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz',
            'test_lbl': 'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz'
        }
    }

    def __init__(self, dataset='mnist', data_dir='./data'):
        self.dataset = dataset
        self.data_dir = os.path.join(data_dir, dataset)
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

    def _download(self, key):
        url = self.URLS[self.dataset][key]
        filename = url.split('/')[-1]
        path = os.path.join(self.data_dir, filename)
        
        if not os.path.exists(path):
            print(f"📡 Requesting {self.dataset} {key} from mirror...")
            opener = urllib.request.build_opener()
            opener.addheaders = [('User-agent', 'Mozilla/5.0')]
            urllib.request.install_opener(opener)
            urllib.request.urlretrieve(url, path)
        return path

    def _load_images(self, path):
        with gzip.open(path, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
        return data.reshape(-1, 784).astype(np.float32) / 255.0

    def _load_labels(self, path):
        with gzip.open(path, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=8)
        return data.astype(np.int64)

    def load(self):
        train_img_path = self._download('train_img')
        train_lbl_path = self._download('train_lbl')
        test_img_path = self._download('test_img')
        test_lbl_path = self._download('test_lbl')

        x_train = self._load_images(train_img_path)
        y_train = self._load_labels(train_lbl_path)
        x_test = self._load_images(test_img_path)
        y_test = self._load_labels(test_lbl_path)

        return (x_train, y_train), (x_test, y_test)

# Backward compatibility
MNISTLoader = lambda: DatasetLoader(dataset='mnist')
