import time
import gc

import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas

from config import DATA_DIR
from src.models import ResNetUNet
from src.utils import load_model, predict, see_plot


imgs_dir = f'{DATA_DIR}/imgs'


def main():
    page_icon = Image.open(f'{imgs_dir}/pytorch_logo_icon.png').resize((24, 24), Image.ANTIALIAS)
    page_icon = page_icon.convert('RGBA')
    st.set_page_config(
        page_title='CCNN Demo App', page_icon=page_icon
    )
    with open('src/form_wo_border.css') as form_style_file:
        st.markdown(f'<style>{form_style_file.read()}</style>', unsafe_allow_html=True)

    st.title('Conditional CNN Demo App')
    st.sidebar.header('Configuration')

    if 'button_id' not in st.session_state:
        st.session_state['button_id'] = ''
    if 'color_to_label' not in st.session_state:
        st.session_state['color_to_label'] = {}

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
    with open(f'{imgs_dir}/Condition_ResUnet_graph.png', 'rb') as file:
        btn = center_col.download_button(
            label='Download graph in png',
            data=file,
            file_name='CCNN.png',
            mime='image/png'
        )
    st.image(f'{imgs_dir}/Condition_ResUnet_graph.png',
             channels='RGB',
             output_format='PNG', )


def full_app():
    # st.sidebar.subheader('Опции')
    st.markdown(
        """
        На этом сайте можно посмотреть, как Conditional CNN двигает рисунки (числа) по 
        выбранному условию.
        
        Для запуска опыта нарисуйте что-нибудь в поле ниже, докрутите кнопки **Выполнить** и нажмите ее. Затем внизу вы увидите как появится картинка с результатом работы нейросети;
        
        Если вы хотите посмотреть историю ваших опытов, кликните на соответствующую галочку в настройках и нажмите **Выполнить** \
        - история появится внизу. 
    
        """
    )
    st.write('')
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
    directions_map = {dir_name: i for i, dir_name in enumerate(directions)}

    model_resunet = load_model(ResNetUNet(3), 'resunet_pl_loss')

    if 'History' not in st.session_state:
        st.session_state['History'] = []
    if 'Executed' not in st.session_state:
        st.session_state['Executed'] = 'initialized'

    with st.form('execute_digit_movement'):
        _col_1, _, _col_2 = st.columns([2, 1, 2])

        # Specify canvas parameters in application
        with _col_2:
            st.subheader('Настройки:')
            drawing_mode = 'freedraw'
            stroke_width = st.slider('Толщина линии рисования: ', 1, 25, 12)

            direction = st.selectbox(
                'Выберите из списка в каком направлении хотите двигать рисунок:',
                directions,
                key='direction_selection')

            direction_shift = st.slider('Величина сдвига в пикселях: ', 1, 30, 12)
            show_history_checkbox = st.checkbox(label='Показывать историю опытов', value=True)

        # Create a canvas component
        with _col_1:
            st.subheader('Нарисуйте любую цифру или фигуру:')
            canvas_result = st_canvas(
                fill_color='rgba(255, 255, 0, 1)',  # Fixed fill color with some opacity
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

        _, center_col_with_button, _ = st.columns([0.75, 1, 0.75])
        with center_col_with_button:
            execute_digit_movement_button = st.form_submit_button(label='Запустить работу нейросети')

        st.subheader('')
        st.subheader('Результат работы нейросети:')

        if st.session_state['Executed'] == 'initialized':
            placeholder = st.empty()
            demo_input_img = Image.open(f'{imgs_dir}/demo_input_img.jpeg')
            demo_prediction_img = Image.open(f'{imgs_dir}/demo_prediction_img.jpeg')
            placeholder.pyplot(see_plot(demo_input_img, demo_prediction_img, size=(4, 4)))
        else:
            placeholder = st.empty()
            with placeholder.container():
                # history is [(elem_1, elem_2, (text)), (...), ...] and we take last pair
                _input_img = st.session_state['History'][-1][0]
                _prediction = st.session_state['History'][-1][1]
                placeholder.pyplot(see_plot(_input_img, _prediction, size=(4, 4)))

    if execute_digit_movement_button:
        if st.session_state['Executed'] == 'initialized':
            st.session_state['Executed'] = 'executed'

        with st.spinner('Двигаю рисунок'):
            image_from_canvas = canvas_result.image_data
            direction = directions_map[direction]
            input_img, prediction = predict(model_resunet, image_from_canvas, direction, direction_shift)
            direction = directions[direction]   # get text repr of direction
            st.session_state['History'].append((input_img, prediction, (direction, direction_shift)))
            placeholder.pyplot(see_plot(input_img, prediction, size=(4, 4)))
            time.sleep(1.1)
        st.success('Done!')

    # Show user's actions history
    if show_history_checkbox:
        if len(st.session_state['History']) > 10:
            del st.session_state['History'][0]
            gc.collect()

        _, center_col, _ = st.columns(3)
        center_col.success('Ваша история действий:')

        _, center_col, _ = st.columns([1, 3, 1])
        with center_col:
            for __input_img, __prediction, __text in st.session_state['History']:
                __text = f'Направление: {__text[0]} со сдвигом {__text[1]}px'
                st.pyplot(see_plot(__input_img, __prediction, text=__text, size=(4, 4)))

