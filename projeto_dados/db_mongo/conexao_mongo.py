'''import os
import json
import datetime
from pymongo import MongoClient
from dotenv import load_dotenv

# Carrega as variáveis de ambiente do .env
load_dotenv()

def conectar_mongo():
    # Usa a URI do MongoDB definida no arquivo .env
    mongo_uri = os.getenv('MONGO_URI')
    cliente = MongoClient(mongo_uri)
    db = cliente['projeto_dados']
    return db

def salvar_no_mongo(data, colecao):
    db = conectar_mongo()
    colecao = db[colecao]
    
    # Criar documento com data de inserção
    documento = {
        "dados": data,
        "data_insercao": datetime.datetime.now()
    }
    
    colecao.insert_one(documento)
    print("Dados inseridos no MongoDB com sucesso!")'''





# db_mongo/conexao_mongo.py
import os
import datetime
from pymongo import MongoClient
from dotenv import load_dotenv

# Carrega as variáveis de ambiente do .env
load_dotenv()

def conectar_mongo():
    """
    Conecta ao MongoDB usando a URI fornecida nas variáveis de ambiente.
    """
    mongo_uri = os.getenv('MONGO_URI')
    if not mongo_uri:
        raise ValueError("A variável de ambiente 'MONGO_URI' não está definida.")
    cliente = MongoClient(mongo_uri)
    db = cliente['projeto_dados']
    return db

def salvar_no_mongo(data, colecao):
    """
    Salva um documento no MongoDB na coleção especificada.

    Args:
        data (list or dict): Dados a serem salvos. Pode ser uma lista de dicionários ou um único dicionário.
        colecao (str): Nome da coleção no MongoDB.
    """
    db = conectar_mongo()
    colecao = db[colecao]
    
    # Adicionar data de inserção
    if isinstance(data, list):
        for entry in data:
            entry['data_insercao'] = datetime.datetime.utcnow()
        colecao.insert_many(data)
    elif isinstance(data, dict):
        data['data_insercao'] = datetime.datetime.utcnow()
        colecao.insert_one(data)
    else:
        raise TypeError("Os dados a serem salvos devem ser uma lista de dicionários ou um dicionário único.")
    
    print(f"Dados inseridos na coleção '{colecao.name}' no MongoDB com sucesso!")
