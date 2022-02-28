import streamlit as st
import matplotlib.pyplot as plt

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T

from config import WEIGHTS_DIR, params


test_aug = T.Compose([
    T.ToTensor(),
    T.Resize((64, 64)),
    T.Normalize((0.5,), (0.5,)),
])


def create_roll_map(d):
    return {
        0: (0, 0, 0),
        1: (0, 0, d),  # 'W'
        2: (0, -d, d),  # 'W+D'
        3: (0, -d, 0),  # 'D'
        4: (0, -d, -d),  # 'S+D'
        5: (0, 0, -d),  # 'S'
        6: (0, d, -d),  # 'S+A'
        7: (0, d, 0),  # 'A'
        8: (0, d, d)  # 'W+A'
    }


@st.cache(allow_output_mutation=True, ttl=5*60)
def load_model(model, model_name):
    model.load_state_dict(torch.load(f'{WEIGHTS_DIR}/{model_name}.pth', map_location=params['DEVICE']))
    model = model.to(params['DEVICE'])
    model.eval()
    return model


@st.cache(ttl=5*60)
def load_consts():
    sp = torch.tensor([0]).unsqueeze(0).to(params['DEVICE'])
    zoom = torch.tensor([1]).unsqueeze(0).to(params['DEVICE'])/15
    return sp, zoom


def predict(model,
            image_from_canvas,
            direction,
            direction_shift):
    np_img = np.array(image_from_canvas[:, :, :3], dtype=np.float32)  # canvas return rgba format

    with np.errstate(invalid='ignore', divide='ignore'):
        tensored_img = test_aug(np_img / np_img.max())
    tensored_img = tensored_img.unsqueeze(0).to(params['DEVICE'])

    direction_shift /= 30.  # norm btw 0 and 1
    direction_shift = torch.as_tensor([direction_shift]).float()
    direction_shift = direction_shift.to(params['DEVICE'])
    _dir = F.one_hot(torch.tensor(direction), num_classes=8).float()  # torch.Size([8])
    _dir = _dir.to(params['DEVICE'])
    _dir = torch.cat((_dir, direction_shift))
    _dir = _dir.unsqueeze(0).to(params['DEVICE'])
    # sp, zoom = load_consts()

    with torch.no_grad():
        prediction = model((tensored_img, _dir))

    source_image = tensored_img.squeeze(0).detach().cpu().numpy()
    source_image = (source_image + 1) / 2
    source_image = np.transpose(source_image, (1, 2, 0))

    prediction = prediction.squeeze(0).detach().cpu().numpy()
    prediction = (prediction + 1) / 2
    prediction = np.transpose(prediction, (1, 2, 0))
    prediction = cv2.cvtColor(prediction, cv2.COLOR_BGR2GRAY)
    return source_image, prediction


def see_plot(target, prediction, text=None, color='gray', size=(4, 4)):
    plt.close('all')
    plt.figure(figsize=size)
    fig, ax = plt.subplots(1, 2, figsize=size)
    fig.patch.set_facecolor('0.8')
    fig.suptitle(text, y=0.75, fontsize=8)

    ax[0].set_title('Ваш рисунок', fontsize=8)
    ax[0].imshow(target, cmap=color)
    ax[0].xaxis.set_ticks([16, 32, 48, 64])
    ax[0].yaxis.set_ticks([16, 32, 48, 64])
    ax[0].grid()
    # ax[0].axis('off')

    ax[1].set_title('Сдвиг рисунка нейросетью', fontsize=8)
    ax[1].imshow(prediction, cmap=color)
    ax[1].xaxis.set_ticks([16, 32, 48, 64])
    ax[1].yaxis.set_ticks([16, 32, 48, 64])
    ax[1].grid()
    # ax[1].axis('off')

    plt.tight_layout()
    return fig

