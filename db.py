from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils import embedding_functions
import pandas as pd

"""
This module was an experiment. The used embedding function works, but the gemini
embeddings just work much better. The gemini embeddings are also faster, because
we're calling an API instead of loading a model.
"""

def load_db():
    """
    Load the database from a file.
    """
    # Placeholder for loading the database
    model = SentenceTransformer("Parallia/Fairly-Multilingual-ModernBERT-Embed-BE-NL")
    return model

def find_similarity(model, query, db):
    """
    Find the most similar item in the database to the query.
    """
    # Placeholder for finding similarity
    # In a real implementation, this would involve querying the database
    # and returning the most similar item based on the model's embeddings.
    # model = load_db()
    # create embedding function using Parallia/Fairly-Multilingual-ModernBERT-Embed-BE
    embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="Parallia/Fairly-Multilingual-ModernBERT-Embed-BE-NL"
    )
    # now create a client using the embedding function
    # client = chromadb.Client()
    client = chromadb.PersistentClient("./chroma_db")

    collection = client.get_or_create_collection("leerdoelen", embedding_function=embedding_function)

    # doelen = df[["leerresultaten", "leerdoelen"]].drop_duplicates()

    # doelen.map(lambda x: x['leerresultaten'] + ": " + x['leerdoelen'], axis=1)

    # read csv file
    df = pd.read_csv("leerdoelen.csv")
    # print(df.head())
    doelen = df[["leerresultaten", "leerdoelen"]].drop_duplicates()

    # concat the columns leerresultaten and leerdoelen with a ": " in between
    # doelen['leerresultaten'] = doelen['leerresultaten'].astype(str)
    # doelen['leerdoelen'] = doelen['leerdoelen'].astype(str)
    doelen['leerresultaten'] = doelen['leerresultaten'].str.cat(doelen['leerdoelen'].astype(str), sep=": ")
    # collection.add(
    #     documents = doelen["leerresultaten"].tolist(),
    #     ids = doelen.index.astype(str).tolist()
    # )

    results = collection.query(
        query_texts = "Complexe design patterns, Multithreading, In optie nog een te bepalen advanced topic objectgeorienteerd programmeren"
    )
    for i in range(len(results["documents"][0])):
        print("Result: ", results["documents"][0][i])
    # print(results)
    return "Most similar item"

    # print(doelen["leerresultaten"].head(), doelen.index)

# print(doelen.apply(lambda x: x['leerresultaten'] + ": " + x['leerdoelen']).head())
find_similarity(None, None, None)
