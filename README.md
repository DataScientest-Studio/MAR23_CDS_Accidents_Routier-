**README**

## Presentation and Installation

This repository contains the code for our project **SARA**, developed during our [Data Scientist training](https://datascientest.com/en/data-scientist-course) at [DataScientest](https://datascientest.com/).

The SARA project (Automobile Analysis and Risk System) aims to predict the severity of road accidents in France by applying Data Science and machine learning techniques to a historical dataset.
The data for this project can be found at the following address: [Data.gouv.fr](https://www.data.gouv.fr/fr/datasets/bases-de-donnees-annuelles-des-accidents-corporels-de-la-circulation-routiere-annees-de-2005-a-2022/)

This project was developed by the following team :

- Cécile Pilon ([GitHub](https://github.com/) / [LinkedIn](http://linkedin.com/))
- Fadimatou Abdoulaye ([GitHub](https://github.com/) / [LinkedIn](http://linkedin.com/))
- Stéphane Maillard ([GitHub](https://github.com/) / [LinkedIn]([https://www.linkedin.com/in/st%C3%A9phane-maillard/])
- Christophe Levra ([GitHub](https://github.com/) / [LinkedIn](http://linkedin.com/))
- Abdoulaye Tall ([GitHub](https://github.com/) / [LinkedIn](http://linkedin.com/))

You can browse and run the [notebooks](./notebooks). 

## Streamlit App

To run the app (be careful with the paths of the files in the app):

```shell
conda create --name my-awesome-streamlit python=3.9
conda activate my-awesome-streamlit
pip install -r requirements.txt
streamlit run app.py
```

The app should then be available at [localhost:8501](http://localhost:8501).
