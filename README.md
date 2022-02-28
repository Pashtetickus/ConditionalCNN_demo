# Conditional image2image model demo
[Here](https://share.streamlit.io/pashtetickus/conditionalcnn_demo/main/app.py) you can see how conditional image2image model moves pictures by the given condition.

This type of neural network models can be adapted and used to simulate various physical processes.


### Usage example:
![Demo CountPages alpha](data/assets/Demo.gif)


Для запуска опыта нарисуйте что-нибудь в специальном поле, докрутите кнопки `Выполнить` и нажмите ее.
Затем внизу вы увидите как появится картинка с результатом работы нейросети.
Если вы хотите посмотреть историю ваших опытов, кликните на соответствующую галочку в настройках и нажмите `Выполнить` - история будет обновляться внизу по мере ваших опытов.


## Structure of this Repo

- [src](src) : main code
- [scripts](scripts) : scripts for data downloading
- [data](data) : downloaded data saved here
- [weights](weights) : learnt weights


## Requirements

- python3
- pip

## Installation & Run
### From source

Clone the repo and change to the project root directory:

```
git clone https://github.com/Pashtetickus/ConditionalCNN_demo.git
cd ConditionalCNN_demo
```


Create venv:

with `conda`
```
conda create -n CCNN_venv python=3.8
conda activate CCNN_venv
```
or with `python`:
```
python -m venv CCNN_venv
source CCNN_venv/bin/activate # Linux
source CCNN_venv/Scripts/activate # Windows
```

Install requirements:

```
python -m pip install -r requirements.txt
```

And run:

```
streamlit run run_demo.py
```
