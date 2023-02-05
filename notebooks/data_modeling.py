from load_database import clean_db
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Ejecutar en caso de requerir poblar la base de datos en postgresql
# path = 'C:/Users/Acer/Desktop/enginner_r5/r5-ds-challenge/data/fraud.csv'
# execute_poblar(path)
    
query = """
WITH fraud AS (
	SELECT monthh, weekofmonth, yearr, fault, policytype,
	TRIM(SPLIT_PART(policytype, '-', 1)) AS policytype_1,
	TRIM(SPLIT_PART(policytype, '-', 2)) AS policytype_2, 
	vehiclecategory, basepolicy, make, accidentarea,
	deductible, driverrating, pastnumberofclaims,
	ageofvehicle, policereportfiled, witnesspresent,
	agenttype, numberofsuppliments, fraudfound_p, age
    
	FROM fraudes 
	WHERE monthh = monthclaimed 
	),
fraud_clean AS (
	SELECT * FROM fraud
	WHERE policytype_1 = vehiclecategory AND age != 0
	),
fraud_comp AS (
	SELECT monthh, weekofmonth, yearr, fault, policytype,
		vehiclecategory, basepolicy, make, accidentarea,
		deductible, driverrating, pastnumberofclaims,
		ageofvehicle, policereportfiled, witnesspresent,
		agenttype, numberofsuppliments, fraudfound_p, age
	FROM fraud_clean
	)

SELECT * 
FROM fraud_comp;
        """
data = clean_db(query)

def modeling(data):
    
    data[['weekofmonth', 'driverrating']] = data[['weekofmonth', 'driverrating']].astype('str') 

    categorical = data.select_dtypes(include='object').columns

    data_category = data[categorical].apply(lambda x: len(x.unique())).reset_index(name='count').sort_values('count', ascending=False)
    print(f'Numero de categorias por variable: \n {data_category}')
    
    return data

data = modeling(data)

def variable_objetivo(data):
    
    conteo = data[['fraudfound_p']].fillna('NA').value_counts().reset_index(name='Count')
    conteo['Total'] = conteo['Count'].sum()
    conteo['Porcentaje'] = conteo['Count'] / conteo['Total'] * 100
    
    sns.set_theme(style='darkgrid')
    sns.countplot(data=data, x='fraudfound_p')
    plt.title('Variable objetivo: Fraudes por reclamaciones')
    plt.xlabel('Fraude')
    
    return print(f'Cantidad de fraudes: \n {conteo}')

# variable_objetivo(data)

def eda(data, column):
    
    try:
        conteo = data[[column,'fraudfound_p']].fillna('NA').value_counts().reset_index(name='Count')#.sort_values(['monthh', 'fraudfound_p'],ascending = [True, True])
        conteo['Total'] = conteo.groupby(column)['Count'].transform('sum')
        conteo['Porcentaje'] = conteo['Count'] / conteo['Total'] * 100
        conteo[conteo['fraudfound_p'] == 1].sort_values(['Porcentaje'], ascending=False)
        
        plt.figure()
        sns.countplot(data=data, x=column, hue='fraudfound_p')
        plt.title(f'Numero de fraudes por {column}')
        plt.xlabel(f'column')
        
        return print(f'Conteo de fraudes por {column}: \n {conteo}')
    
    except Exception as e:
        print(f'La columna especificada "{column}" no existe en el DataFrame')

# eda(data, 'witnesspresent')

import category_encoders as ce
import pickle

def encoder(data):

    encoder=ce.OneHotEncoder(handle_unknown='ignore',return_df=True,use_cat_names=True,drop_invariant=True)
    data_encoded = encoder.fit(data)
    data_encoded
    
    data_encoder = data_encoded.transform(data)
    
    data_encoded.get_params
    
    with open('encode.pkl', 'wb') as f:
        pickle.dump(data_encoder, f)
        


"""
Segun la limpieza que se le realizo a la data, como eliminacion de edades en cero, registros los cuales
no coincidian con la informacion de policytype con vehiclecategory y basepolicy queda un otal de 7378
registros, los cuales el 7.91% corresponden a fraudes
"""

"""
Tomando en cuenta el analisis exploratorio de los datos, se debe conciderar las variables seleccionadas
ya que brindan informacion relevante como:
    * EL mayor porcentaje de fraudes se presena en los meses de mayo y agosto.
    * Se observa que la mayoria de fraudes han ocurrido en las categorias 
        Sedan - All Perils y Sedan - Collision
    * Hay un porcentaje muy bajo para los fraudes en los cuales hay testigo
    * Se puede decir que los fraudes no son realizados por personal de la empresa
    * Se realizo el encoding de los datos, el cual puede ser usado en producción, para la 
        transformación de la data extraida via postgresql
"""
