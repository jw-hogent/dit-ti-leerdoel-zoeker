import streamlit as st
import pandas as pd 
from gemini import get_db, populate_db, query_db

st.title("Leerdoel zoeker")

leerdoelen = pd.read_csv("leerdoelen.csv")
# print(leerdoelen.head())
leerdoelen_orig = leerdoelen.copy()

nieuwe_leerdoelen = pd.read_csv("nieuwe_leerdoelen.csv", sep=";")
# print(nieuwe_leerdoelen.head())
# print(nieuwe_leerdoelen.shape)

# check that leerdoelen is unique based on LDcode
# print(leerdoelen.shape)
leerdoelen = leerdoelen.drop_duplicates(subset=["LDcode"])
# print(leerdoelen.shape)

# check that nieuwe leerdoelen is unique based on LDcode
nieuwe_leerdoelen = nieuwe_leerdoelen.drop_duplicates(subset=["LDcode"])
# print(nieuwe_leerdoelen.shape)

# create a new dataframe with LDcode, leerresultaten and leerdoelen
alle_doelen = nieuwe_leerdoelen[["LDcode", "leerresultaten", "leerdoelen", "LRcode"]].drop_duplicates()
# concatenate this df with leerdoelen
alle_doelen = pd.concat([alle_doelen, leerdoelen[["LDcode", "leerresultaten", "leerdoelen", "LRcode"]].drop_duplicates()])

alle_doelen = alle_doelen.drop_duplicates(subset=["LDcode"])
# print(alle_doelen.shape)
# concatenate the columns leerresultaten and leerdoelen with a ". " in between
alle_doelen['text'] = alle_doelen['leerresultaten'].str.cat(alle_doelen['leerdoelen'].astype(str), sep=". ")

nieuwe_vakken = dict()
for i in range(nieuwe_leerdoelen.shape[0]):
    nieuwe_vakken[nieuwe_leerdoelen["LDcode"].iloc[i]] = nieuwe_leerdoelen["olods"].iloc[i]
oude_vakken = dict()
oude_vakken = leerdoelen_orig.groupby(["LDcode"])[["OLOD"]].agg(lambda x: ", ".join(x)).to_dict()['OLOD']
# print(oude_vakken)
# print(nieuwe_vakken)

leerresultaat_mapping = pd.read_csv("leerresultaat_mapping.csv", sep=",")

db = get_db()

# is already populated, to reload: delete file and uncomment
# populate_db(db, alle_doelen['text'].tolist(), alle_doelen['LDcode'].astype(str).tolist())

def search_data(query):
    results = query_db(db, query)
    # for i in range(len(results["documents"][0])):
    #     print(f"key: {results['ids'][0][i]}, Result: {results['documents'][0][i]}")
    return alle_doelen[alle_doelen['LDcode'].isin(results['ids'][0])].to_dict(orient="records")

# Simple UI: text input and a search button.
query = st.text_input("Beschrijf wat je zoekt:", "")

if not query.isspace() and query != "":
    # Get search results from your function.
    results = search_data(query)
    
    if results:
        # st.write(f"Found {len(results)} result(s).")
        for result in results:
            # print(result)
            if result['LRcode'] not in leerresultaat_mapping['LRcode'].values:
                continue
            # Render each result as a card using a container.
            with st.container():
                resultaat = leerresultaat_mapping[leerresultaat_mapping['LRcode'] == result['LRcode']]
                st.markdown(f"### Leerlijn: {resultaat['leerlijn'].values[0]}")
                st.markdown(f"**Tags:** {resultaat['tags'].values[0]}")
                st.markdown(f"**{result['LRcode']}:** {result['leerresultaten']}")
                st.markdown(f"**{result['LDcode']}:** {result['leerdoelen']}")
                # st.markdown(f"**Level:** {result['level']}")
                # st.markdown(f"**Subjects:** {result['subjects']}")
                print(f"result['LDcode'] {result['LDcode']}")
                if result['LDcode'] in oude_vakken:
                    print(f"result['LDcode'] in oude_vakken {result['LDcode']}")
                    st.markdown(f"**Used in:** {oude_vakken[result['LDcode']]}")
                if result['LDcode'] in nieuwe_vakken:
                    print(f"result['LDcode'] in nieuwe_vakken {result['LDcode']}")
                    st.markdown(f"**Suggested OLODs:** {nieuwe_vakken[result['LDcode']]}")
                
                st.write("---")
    else:
        st.write("No results found.")
