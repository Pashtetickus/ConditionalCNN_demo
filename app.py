import time
import streamlit as st
import streamlit.components.v1 as components
from streamlit_drawable_canvas import st_canvas

import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as T
import torch.nn.functional as F
from torch import nn
from torchvision import models
from PIL import Image

batch_size = 32
params = {
    'DEVICE': 'cuda:0' if torch.cuda.is_available() else 'cpu',
}


def see_plot(target, prediction, text=None, color='gray', size=(4, 4)):
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


def convrelu(in_channels, out_channels, kernel, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )

class ResNetUNet(nn.Module):
    def __init__(self, n_class, DEBUG=False, _tensorboard=False):
        super().__init__()
        self.DEBUG = DEBUG
        self._tensorboard = _tensorboard

        self.base_model = models.resnet18(pretrained=True)

        self.base_layers = list(self.base_model.children())

        self.layer0 = nn.Sequential(*self.base_layers[:3])  # size=(N, 64, x.H/2, x.W/2)
        self.layer0_1x1 = convrelu(64, 64, 1, 0)
        self.layer1 = nn.Sequential(*self.base_layers[3:5])  # size=(N, 64, x.H/4, x.W/4)
        self.layer1_1x1 = convrelu(64, 64, 1, 0)
        self.layer2 = self.base_layers[5]  # size=(N, 128, x.H/8, x.W/8)
        self.layer2_1x1 = convrelu(128, 128, 1, 0)
        self.layer3 = self.base_layers[6]  # size=(N, 256, x.H/16, x.W/16)
        self.layer3_1x1 = convrelu(256, 256, 1, 0)
        self.layer4 = self.base_layers[7]  # size=(N, 512, x.H/32, x.W/32)
        self.layer4_1x1 = convrelu(512, 512, 1, 0)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv_up3 = convrelu(256 + 512, 512, 3, 1)
        self.conv_up2 = convrelu(128 + 512, 256, 3, 1)
        self.conv_up1 = convrelu(64 + 256, 256, 3, 1)
        self.conv_up0 = convrelu(64 + 256, 128, 3, 1)

        self.conv_original_size0 = convrelu(3, 64, 3, 1)
        self.conv_original_size1 = convrelu(64, 64, 3, 1)
        self.conv_original_size2 = convrelu(64 + 128, 64, 3, 1)
        self.dropout = nn.Dropout(0.5)
        self.conv_last = nn.Conv2d(64, n_class, 1)
        self.act_last = nn.Tanh()
        self.support_conv1 = nn.Conv2d(9, 512, 1)  # (bath, 9) --> (batch, 512)

    # for tensorboard: (idk how to write it better)
    # def forward(self, inp_0, inp_1):
    def forward(self, inp):
        if self.DEBUG:
            # summary can't work with multiple inputs so we create them for him
            if not self._tensorboard:
                inp_0 = torch.rand(batch_size, 3, 64, 64)
                inp_1 = torch.rand(batch_size, 9)
            x_original = self.conv_original_size0(inp_0)
        else:
            x_original = self.conv_original_size0(inp[0])

        x_original = self.conv_original_size1(x_original)

        if self.DEBUG:
            layer0 = self.layer0(inp_0)
        else:
            layer0 = self.layer0(inp[0])

        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

        if self.DEBUG:
            cond = self.support_conv1(
                torch.unsqueeze(torch.unsqueeze(inp_1, 2), 2))  # ([batch, 9]) --> Size([9, 512, 1, 1])
        else:
            cond = self.support_conv1(
                torch.unsqueeze(torch.unsqueeze(inp[1], 2), 2))  # ([batch, 9]) --> Size([9, 512, 1, 1])

        layer4 = self.layer4_1x1(layer4 + cond)

        x = self.upsample(layer4)
        layer3 = self.layer3_1x1(layer3)
        x = torch.cat([x, layer3], dim=1)
        x = self.conv_up3(x)

        x = self.upsample(x)
        layer2 = self.layer2_1x1(layer2)
        x = torch.cat([x, layer2], dim=1)
        x = self.conv_up2(x)

        x = self.upsample(x)
        layer1 = self.layer1_1x1(layer1)
        x = torch.cat([x, layer1], dim=1)
        x = self.conv_up1(x)

        x = self.upsample(x)
        layer0 = self.layer0_1x1(layer0)
        x = torch.cat([x, layer0], dim=1)
        x = self.conv_up0(x)

        x = self.upsample(x)
        x = torch.cat([x, x_original], dim=1)
        x = self.conv_original_size2(x)

        x = self.dropout(x)
        out = self.conv_last(x)
        out = self.act_last(out)

        return out


model_test = ResNetUNet(n_class=3, DEBUG=False)
model_test.load_state_dict(torch.load('resunet_mse.pth', map_location=torch.device('cpu')))
model_test = model_test.to('cpu')
model_test = model_test.eval()


test_aug = T.Compose([
    T.ToTensor(),
    T.Resize((64, 64)),
    T.Normalize((0.5), (0.5)),
])


def main():
    if "button_id" not in st.session_state:
        st.session_state["button_id"] = ""
    if "color_to_label" not in st.session_state:
        st.session_state["color_to_label"] = {}
    PAGES = {
        'Conditional-ResUnet example': full_app,
        'Граф модели (Tensorboard)': show_graph,
    }
    page = st.sidebar.selectbox('Page:', options=list(PAGES.keys()), key='PAGE_selection')

    PAGES[page]()

    with st.sidebar:
        st.markdown("---")
        st.markdown(
            '<h6>Made in &nbsp<img src="https://streamlit.io/images/brand/streamlit-mark-color.png" alt="Streamlit logo" height="16">&nbsp '
            'with a source code from <a href="https://github.com/andfanilo/streamlit-drawable-canvas">@andfanilo</a> by <a href="https://vk.com/danilov_ps">@p.danilov</a></h6>',
            unsafe_allow_html=True,
        )


def show_graph():
    st.markdown(
        """            
    Посмотреть на граф можно тут: [Tensorboard](https://tensorboard.dev/experiment/0Yg6rT3kSd2Jf9j6B9cyew/#graphs&run=Condition_ResUnet_test)

    В боковом меню *Tensorboard* нужно будет выключить тумблер *Auto-extract high-degree nodes* для корректного отображения.
    """
    )
    st.write('Это скриншот графа для примера (можно скачать по кнопке ниже):')
    _col_1, center_col, _col_2 = st.columns(3)
    with open('img/Condition_ResUnet_graph.png', 'rb') as file:
        btn = center_col.download_button(
            label='Download graph in png',
            data=file,
            file_name='CCNN.png',
            mime='image/png'
        )
    st.image('img/Condition_ResUnet_graph.png',
             channels='RGB',
             output_format='PNG', )


def full_app():
    st.markdown(
        """
    На этом сайте можно посмотреть, как нейросетка двигает рисунки (числа) по 
    выбранному условию.
    
    Для запуска опыта нарисуйте что-нибудь в поле ниже, после кликните на пустое место на странице и
    затем можете управлять направлением движения рисунка через WASD или стрелки. Чем больше вы кликните, например на **S+D**,
    тем выше и выше будет джвигаться рисунок. При некоторой сноровке (чуть зажать 2 клавиши или совсем одновременно)
    можно двигаться по диагонали.
    """
    )
    st.write('')

    directions = ['вниз',
                  'вниз и влево',
                  'влево',
                  'вверх и влево',
                  'вверх',
                  'вверх и вправо',
                  'вправо',
                  'вниз и вправо',
                  ]
    directions_map = {v: i for i, v in enumerate(directions)}

    if 'History' not in st.session_state:
        st.session_state['History'] = []
    if 'Executed' not in st.session_state:
        st.session_state['Executed'] = 'initialized'

    def create_roll_map(d):
        dir = {0: (0, 0, 0),
               1: (0, 0, d),      # 'W'
               2: (0, -d, d),     # 'W+D'
               3: (0, -d, 0),     # 'D'
               4: (0, -d, -d),    # 'S+D'
               5: (0, 0, -d),     # 'S'
               6: (0, d, -d),     # 'S+A'
               7: (0, d, 0),      # 'A'
               8: (0, d, d)       # 'W+A'
               }
        return dir

    # !===== PyTorch part =====! #
    def predict(image_from_canvas,
                direction,
                direction_shift):
        # np_img = np.array(image_from_canvas[:, :, :3], dtype=np.float32)  # canvas return rgba format
        if image_from_canvas.shape[-1] > 3:
            np_img = np.array(image_from_canvas[:, :, :3], dtype=np.float32)
        else:
            np_img = image_from_canvas

        with np.errstate(invalid='ignore', divide='ignore'):
            tensored_img = test_aug(np_img / np_img.max())
        tensored_img = tensored_img.unsqueeze(0).to(params['DEVICE'])
        direction = directions_map[direction]
        roll_map = create_roll_map(direction_shift)

        direction_shift /= 30.  # norm btw 0 and 1
        direction_shift = torch.as_tensor([direction_shift]).float()
        _dir = F.one_hot(torch.tensor(direction), num_classes=8).float()  # torch.Size([8])
        _dir = torch.cat((_dir, direction_shift))
        _dir = _dir.unsqueeze(0).to(params['DEVICE'])

        with torch.no_grad():
            prediction = model_test((tensored_img, _dir))

        source_image = torch.roll(tensored_img.squeeze(0), roll_map[direction + 1], (0, 2, 1))
        source_image = source_image.detach().cpu().numpy()
        source_image = (source_image + 1) / 2
        source_image = np.transpose(source_image, (1, 2, 0))

        prediction = prediction.squeeze(0).detach().cpu().numpy()
        prediction = (prediction + 1) / 2
        prediction = np.transpose(prediction, (1, 2, 0))
        return source_image, prediction

    # !===== PyTorch part ENDS =====! #


    _col_1, _, _col_2 = st.columns([2, 1, 2])

    # Specify canvas parameters in application
    with _col_2:
        st.subheader('Настройки:')
        drawing_mode = 'freedraw'
        stroke_width = st.slider('Толщина линии рисования: ', 1, 25, 8)
        direction_shift = st.slider('Величина сдвига в пикселях: ', 1, 30, 12)

    # Create a canvas component
    with _col_1:
        st.subheader("В этом поле вы можете рисовать:")
        canvas_result = st_canvas(
            fill_color="rgba(255, 255, 0, 1)",  # Fixed fill color with some opacity
            stroke_width=stroke_width,
            stroke_color='#fff',
            background_color='#000',
            background_image=None,
            update_streamlit=True,
            height=256,
            width=256,
            drawing_mode=drawing_mode,
            display_toolbar=True,
            key="full_app",
        )
    #
    up_dir = st.sidebar.button('W')
    up_right_dir = st.sidebar.button('WD')
    right_dir = st.sidebar.button('D')
    down_right_dir = st.sidebar.button('SD')
    down_dir = st.sidebar.button('S')
    down_left_dir = st.sidebar.button('SA')
    left_dir = st.sidebar.button('A')
    up_left_dir = st.sidebar.button('WA')

    # up_dir = st.button('W')
    # up_right_dir = st.button('WD')
    # right_dir = st.button('D')
    # down_right_dir = st.button('SD')
    # down_dir = st.button('S')
    # down_left_dir = st.button('SA')
    # left_dir = st.button('A')
    # up_left_dir = st.button('WA')

    st.subheader('Ниже можете видеть результат работы нейросети: ')

    if st.session_state['Executed'] == 'initialized':
        placeholder = st.empty()
        demo_input_img = Image.open('img/demo_input_img.jpeg')
        demo_prediction_img = Image.open('img/demo_prediction_img.jpeg')
        placeholder.pyplot(see_plot(demo_input_img, demo_prediction_img, size=(4, 4)))
    else:
        placeholder = st.empty()
        with placeholder.container():
            # history is [(elem_1, elem_2, (text)), (...), ...] and we take last pair
            _input_img = st.session_state['History'][0]
            _prediction = st.session_state['History'][1]
            placeholder.pyplot(see_plot(_input_img, _prediction, size=(4, 4)))

    direction_placeholder = st.empty()
    directions_buttons_activations = np.array([
        down_dir,
        down_left_dir,
        left_dir,
        up_left_dir,
        up_dir,
        up_right_dir,
        right_dir,
        down_right_dir,
    ])

    # get index of True element (direction button activation)
    direction_activation = np.nonzero(directions_buttons_activations)[0]
    if direction_activation.size != 0:
        direction = directions[direction_activation[0]]
        # print(direction)
        direction_placeholder.info(f'полученное направление: {direction}')
        with st.spinner('Двигаю рисунок'):
            image_from_canvas = canvas_result.image_data

            if st.session_state['Executed'] == 'initialized':
                input_img, prediction = predict(image_from_canvas, direction, direction_shift)
                # time.sleep(0.05)
                placeholder.pyplot(see_plot(input_img, prediction, size=(4, 4)))
                st.session_state['History'] = (input_img, prediction, image_from_canvas)
                st.session_state['Executed'] = 'executed'

            # if we didn't update canvas
            elif np.array_equal(st.session_state['History'][-1], image_from_canvas):
                input_img, prediction = predict(st.session_state['History'][0], direction, direction_shift)
                placeholder.pyplot(see_plot(input_img, prediction, size=(4, 4)))
                st.session_state['History'] = (input_img, prediction, image_from_canvas)
            else:
                input_img, prediction = predict(image_from_canvas, direction, direction_shift)
                placeholder.pyplot(see_plot(input_img, prediction, size=(4, 4)))
                st.session_state['History'] = (input_img, prediction, image_from_canvas)


if __name__ == "__main__":
    page_icon = Image.open('img/pytorch_logo_icon_170820.png').resize((24, 24), Image.ANTIALIAS)
    page_icon = page_icon.convert("RGBA")
    st.set_page_config(
        page_title="CCNN Demo App", page_icon=page_icon, initial_sidebar_state='collapsed'
    )
    with open('st_form_wo_border.css') as form_style_file:
        st.markdown(f'<style>{form_style_file.read()}</style>', unsafe_allow_html=True)

    st.title("Conditional CNN wasd control (POC)")

    st.sidebar.header("Configuration")
    main()

    components.html(
        """
    <script>
    const doc = window.parent.document;
    buttons = Array.from(doc.querySelectorAll('button[kind=primary]'));

    var up_dir = false,
        up_right_dir = false,
        right_dir = false,
        down_right_dir = false,
        down_dir = false,
        down_left_dir = false,
        left_dir = false,
        up_left_dir = false

    const W = buttons.find(el => el.innerText === 'W');
    const A = buttons.find(el => el.innerText === 'A');
    const S = buttons.find(el => el.innerText === 'S');
    const D = buttons.find(el => el.innerText === 'D');
    const WA = buttons.find(el => el.innerText === 'WA');
    const WD = buttons.find(el => el.innerText === 'WD');
    const SD = buttons.find(el => el.innerText === 'SD');
    const SA = buttons.find(el => el.innerText === 'SA');

    doc.addEventListener('keydown', press)
    function press(e) {
      if (e.keyCode === 38 /* up */ || e.keyCode === 87 /* w */){
        up_dir = true
      }
      if (e.keyCode === 39 /* right */ || e.keyCode === 68 /* d */){
        right_dir = true
      }
      if (e.keyCode === 40 /* down */ || e.keyCode === 83 /* s */){
        down_dir = true
      }
      if (e.keyCode === 37 /* left */ || e.keyCode === 65 /* a */){
        left_dir = true
      }
    }

    doc.addEventListener('keyup', release)
    function release(e){
      if (e.keyCode === 38 /* up */ || e.keyCode === 87 /* w */){
        up_dir = false
      }
      if (e.keyCode === 39 /* right */ || e.keyCode === 68 /* d */){
        right_dir = false
      }
      if (e.keyCode === 40 /* down */ || e.keyCode === 83 /* s */){
        down_dir = false
      }
      if (e.keyCode === 37 /* left */ || e.keyCode === 65 /* a */){
        left_dir = false
      }
    }

    function gameLoop(){
        if (up_dir){
          W.click();
        }
        if (right_dir){
          D.click();
        }
        if (down_dir){
          S.click();
        }
        if (left_dir){
          A.click();
        }
        if (up_dir && right_dir){
          WD.click();
        }
        if (right_dir && down_dir){
          SD.click();
        }
        if (down_dir && left_dir){
          SA.click();
        }
        if (up_dir && left_dir){
          WA.click();
        }
        window.requestAnimationFrame(gameLoop)
    }
    //setTimeout(gameLoop, 1000);
    window.requestAnimationFrame(gameLoop)

    </script>
    """,
        height=0,
        width=0,
    )
