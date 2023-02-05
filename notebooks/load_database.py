import pandas as pd 
from sqlalchemy import create_engine
import seaborn as sns

def connect_db(username='postgres', password='1020', host='localhost', port='5433', database='prueba_r5'):
    
    try:
        engine = create_engine(f'postgresql://{username}:{password}@{host}:{port}/{database}')
        conn = engine.connect()
        
        if conn:
            print('Conectado correctamente a la base de datos')
            
    except Exception as e:
        print(f'Error al conectar a la base de datos: {e}')
        
    return conn, engine


def load_csv(file):
    
    try:
        return pd.read_csv(file)
    
    except Exception as e:
        print(f'Error al cargar archivo csv: {e}')
        return None

def poblar_db(path_csv, engine):

    df = load_csv(path_csv)

    df.columns = df.columns.str.lower()
    
    df.to_sql('fraudes', engine, if_exists='replace', index=False)
    
    return print('Se cargaron {} registros en la base de datos'.format(len(df)))


def execute_poblar(path):
    conn, engine = connect_db()
    poblar_db(path, engine)
    
    if conn.close:
        print('Desconectado correctamente de la db')
        
# Liempieza de la base datos
def clean_db(query):
    conn, engine = connect_db()
    try:
        query_sql = query
        data = pd.read_sql(query_sql, engine)
        print('Limpieza de la base de datos satisfactoria')
        conn.close
        print("Desconectado de la base de datos")
        return data
    
    except Exception as e:
        return print('Error en la sintaxis de SQL')
        