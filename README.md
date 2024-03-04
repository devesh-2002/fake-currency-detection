# Fake Currency Detection
This project takes an image of the currency and checks if the currency uploaded is Fake or Real. VGG16 Model is used here as a pre-trained CNN Model, However it is fine tuned on a different dataset.

## Tech Stack
1. React
2. Flask
3. Jupyter-Notebook

## Running the project 
1. Fork and Clone the forked repository.
2. Move in the frontend folder, type below command to install the packages.
```
yarn
```
3. Run the frontend by the command :
```
yarn run dev
```
4. Move to the flask folder, and create an environment.
```
virtualenv env
```
5. Activate the environment.
```
env/Scripts/activate
```
6. Install the libraries mentioned in the requirements.txt in the activated environment.
```
pip install -r requirements.txt
```
7. Run the below command to run the flask app :
```
python app.py
```

### Screenshot 
<img width="712" alt="image" src="https://github.com/devesh-2002/fake-currency-detection/assets/79015420/a609c96d-60bf-44f7-b874-b0627376891f">

