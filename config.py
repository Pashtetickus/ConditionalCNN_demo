from pathlib import Path
from torch.cuda import is_available as cuda_is_available


ROOT_DIR = Path(__file__).resolve(strict=True).parent
DATA_DIR = ROOT_DIR / 'data'
WEIGHTS_DIR = ROOT_DIR / 'weights'


params = {
    'DEVICE': 'cuda:0' if cuda_is_available() else 'cpu',
}

