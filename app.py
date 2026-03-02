import pickle as pkl
import pandas as pd
import streamlit as st

with open('house_price_pipeline.pkl', 'rb') as f:
    loaded_pipe = pkl.load(f)

st.header('House Prediction Price')
st.subheader('Enter house details to predict sale price')

MSSubClass = st.number_input('MSSubClass', min_value=20, max_value=190, value=60)
LotArea = st.number_input('LotArea', value=8000)
ConstructedArea = st.number_input('ConstructedArea', value=1500)
LotFrontage = st.number_input('LotFrontage', value=60)
GarageArea = st.number_input('GarageArea', value=400)
HouseAge = st.number_input('HouseAge', value=20)
RemodAge = st.number_input('RemodAge', value=5)
TotalWashrooms = st.number_input('TotalWashrooms', value=2)
TotRmsAbvGrd = st.number_input('TotRmsAbvGrd', value=6)
GarageCars = st.number_input('GarageCars', value=2)
Fireplaces = st.number_input('Fireplaces', value=1)
OverallQual = st.slider('OverallQual', 1, 10, 5)
OverallCond = st.slider('OverallCond', 1, 10, 5)
IsRemodeled = st.checkbox('IsRemodeled')

BldgType = st.selectbox('BldgType', ['2fmCon','TwnhsE','Duplex','1Fam'])
HouseStyle = st.selectbox('HouseStyle', ['1.5Unf','2.5Unf','SFoyer','SLvl','1.5Fin','1Story','2Story'])
Foundation = st.selectbox('Foundation', ['Wood','Stone','Slab','BrkTil','CBlock','PConc'])
MSZoning = st.selectbox('MSZoning', ["RH","RM","FV","RL"])
Street = st.selectbox('Street', ['Grvl','Pave'])
Neighborhood = st.selectbox('Neighborhood', ["IDOTRR","MeadowV","BrDale","NPkVill","OldTown","Sawyer","BrkSide",
                                             "NAmes","Edwards","SawyerW","Mitchel","SWISU","CollgCr","Gilbert",
                                             "NWAmes","Blmngtn","ClearCr","Somerst","Timber","NoRidge","Crawfor",
                                             "StoneBr","Veenker","NridgHt"])
KitchenQual = st.selectbox('KitchenQual', ['Fa','TA','Gd','Ex'])
ExterQual = st.selectbox('ExterQual', ['Fa','TA','Gd','Ex'])
Condition1 = st.selectbox('Condition1', ["RRNn","RRNe","RRAe","RRAn","Artery","Feedr","Norm","PosN","PosA"])

input_df = pd.DataFrame({
    'MSSubClass':[MSSubClass],
    'LotArea':[LotArea],
    'ConstructedArea':[ConstructedArea],
    'LotFrontage':[LotFrontage],
    'GarageArea':[GarageArea],
    'HouseAge':[HouseAge],
    'RemodAge':[RemodAge],
    'TotalWashrooms':[TotalWashrooms],
    'TotRmsAbvGrd':[TotRmsAbvGrd],
    'GarageCars':[GarageCars],
    'Fireplaces':[Fireplaces],
    'OverallQual':[OverallQual],
    'OverallCond':[OverallCond],
    'IsRemodeled':[int(IsRemodeled)],
    'BldgType':[BldgType],
    'HouseStyle':[HouseStyle],
    'Foundation':[Foundation],
    'MSZoning':[MSZoning],
    'Street':[Street],
    'Neighborhood':[Neighborhood],
    'KitchenQual':[KitchenQual],
    'ExterQual':[ExterQual],
    'Condition1':[Condition1]
})

if st.button('Predict Sale Price'):
    prediction = loaded_pipe.predict(input_df)
    st.success(f'Predicted Sale Price: ${prediction[0]:,.2f}')