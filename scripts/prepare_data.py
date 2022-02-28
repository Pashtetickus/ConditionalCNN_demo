import gdown
from config import WEIGHTS_DIR


def load_weights(weights_dir=WEIGHTS_DIR) -> None:
    gdown.cached_download(
        path=str(weights_dir / 'resunet_pl_loss.pth'),
        md5='0d7cfd1f45453c817765b594ad831d66',
        id='1GPdY2_CIFQ3W4DQeC-0SLlrOu_rnMOR8'
    )


if __name__ == '__main__':
    load_weights(WEIGHTS_DIR)
