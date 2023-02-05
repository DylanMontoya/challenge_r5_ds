# PRUEBA DATA SCIENTIST R5

### *Objetivo ppal: Reducir perdidas por fraude en las reclamaciones de siniestro*

## Porqué es importante el modelo en el negocio

Los principales proyectos de R5 vienen encaminados al sector vehículos. Por lo que para la empresa es de gran importancia un modelo de confianza para realizar la prediccion de fraudes, ya que de acuerdo a este se puden tomar decisiones importantes para R5, este estudio podría reducir costos de acuerdo al cubrimiento del seguro por eventos que no fueron accidentes, si no que posiblemente estan siendo provocados intencional mente por el cliente.

Para ellos se desarrolló un modelo de machine learing, capaz de predecir si el evento es un posible fraude. Esto lo realiza el modelo una vez el resgistro esté en la base de datos.

Con este estudio, la empresa puede apoyarse desde el momento que es reportado el accidente, para así enfocarsese en la investigación de dicho evento, ahorrando costos y ya que no se enfocaria en los casos que no son fraudes y se enfocaría en los eventos que tengan mayor probabilidad. Con un evento detectado como fraudulento, la empresa tomaria las acciones correspondientes entrando a relizar un analisis enfocado a las variables seleccionadas, ya que la caracterizacion de estás se realizó eusastivamente con un estudio previo al campo, es decir, se uso una metodología ASUM - DM, la cual juega un papel importante dentro del modelo, ya que ayuda a comprender muy bien el negocio, nos permite hacer un enfoque analitico y realizar una selección adeacuada de los datos para desarrollar un modelo confiable que al evaluarlo sea de confiar al momento de realizar un despliegue en producción.

Por ultimo, este modelo de machine learning se podría montar a producción mediante una API, la cual permita empaquetar el modelo ya sea con alguno de los diferentes FrameWorks como Flask, FastAPI o Django, con el fin de acceder al modelo localmente desde nuestro navegador de internet.
Otra alternativa seria desplegar el modelo con TensorFlow, PyTorch o con servicios de la nueve (AWS, Azure o GCP).

De manera adicional juntando el tema de la modelación y el negocio se pueden agregar reglas del negocio desde la experiencia de las áreas de la entidad vinculadas al sector de vehículos, mejorando la precisión del modelo y disminuyendo la cantidad de falsos positivos.


## Instrucciones del proyecto:

A continuación, se brinda una breve instrucción para entender y ejecutar correctamente el ptoyecto:

1. En [notebooks](./notebooks) estan los scripts para el tratamiento de datos:
    * load_database: permite poblar una base de datos creada en postgresql mediante un archivo csv.
    * data_modeling: permite modelar los datos de acuerdo al enfoque analitico, pues allí se tiene en cuenta las variables de interes, a su vez permite realizar un análisis exploratorio de los datos con la función EDA.
    * r5_ds_model: este script contiene el codigo de todo el modelo completo, pues posteriormente a este se divide en los archivos train.py y predict.py.
    * query_over: es un script de sql, el cual brinda la salida correspondiante a:
    [salida_entregada](./data/salida_entregada.png).

2. el archivo [requerimientos]() contiene las librerias y versiones que se utilizaron en el proyecto.

3. Para la base de datos se requiere de una conexión dicha función esta en load_database, permitiendo obtener toda la informacion para el análisis (la base de datos se debe crear directamente desde postgressql y esta es poblada directamente desde python).
[base de datos](./data/db_psql.png)

4. Análisis Descriptivo, se puede encontrar el analisis descriptivo y conclusiones en notebooks/load_database y notebooks/data_modeling.

5. Encoding, se puede observar las variables transformadas las cuales ingresan al modelo en direcctorio raíz encode.pkl el cual ya se encuentra listo para ser utilizado.

6. El modelo dearrollado se encuentra en models/Model_Fraude.pkl. Listo para ser utilizado nuevamente o realizar despliegue meidante una API o TensorFlow - Pytorch.

7. Archivos solicitados train y predict se encuentran directamente en la ruta principal.

