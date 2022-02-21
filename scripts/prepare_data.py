import gdown
from config import WEIGHTS_DIR


def load_weights(weights_dir=WEIGHTS_DIR) -> None:
    gdown.cached_download(
        path=str(weights_dir / 'resunet_mse.pth'),
        md5='1d5709026342ac80ea629f5ce0d084a1',
        id='1ofqsd6Mhkeul8-MwvTOOeokXb_Ky4XSf'
    )


if __name__ == '__main__':
    load_weights(WEIGHTS_DIR)
