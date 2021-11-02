import requests
import streamlit as st
import pandas as pd
import json
from requests.exceptions import Timeout, SSLError
from bs4 import BeautifulSoup
from ibm_watson import NaturalLanguageUnderstandingV1, ApiException
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
from ibm_watson.natural_language_understanding_v1 import Features, EntitiesOptions, KeywordsOptions
import base64

# Dictionaries

serp = {
	'urls': [],
	'titles': [],
	'meta_desc': [],
	'h1': [],
	'h2': [],
	'h3': []
}

cats_kw = {
	"Keyword Text": [],
	"Keyword Count": [],
	"Keyword Relevance": []
}
cats_ent = {
	"Entity Text": [],
	"Entity Count": [],
	"Entity Type": [],
	"Entity Relevance": []
}

# Header content

st.title("Welcome to WatSERP!")

ibm_api_key = st.text_input("Enter API key for IBM Watson")

serpstack_api_key = st.text_input("Enter API key for Serpstack")

keyword = st.text_input("Enter keyword").lower()

submit = st.button('Submit')

# Watson credentials

authenticator = IAMAuthenticator(ibm_api_key)
natural_language_understanding = NaturalLanguageUnderstandingV1(
	version='2021-03-25',
	authenticator=authenticator)

natural_language_understanding.set_service_url('https://api.eu-gb.natural-language-understanding.watson.cloud.ibm.com/instances/99fba91f-2f7c-4c9a-a04b-7f25e1cd239b')

# Serpstack parameters

params = {
			'access_key': serpstack_api_key,
			'query': keyword,
			'gl': 'uk'
}

# Main code for script

if submit:

	st.markdown('Processing, please wait...')

	google_search_url = 'http://www.google.com/search?q=' + keyword.replace(' ', '+')

	st.write(google_search_url)

	api_result = requests.get('http://api.serpstack.com/search', params, verify=False)

	api_response = api_result.json()

	org = api_response.get("organic_results")

	for results in range(0,len(org)):
		link = (org[results]['url'])
		serp['urls'].append(link)
		serp['titles'].append(org[results]['title'])
		serp['meta_desc'].append(org[results]['snippet'])
		try:
			page = requests.get(link, timeout=10)
		except Timeout:
			st.error(f"The link {link} timed out!")
		except SSLError:
			st.error(f"The link {link} has an SSL error!")
		else:
			if page.status_code in [400, 401, 402, 404, 407, 408, 429, 500, 502, 503, 504]:
				continue
		soup = BeautifulSoup(page.content, "html.parser", from_encoding="iso-8859-1")
		for headings in soup.find_all(["h1", "h2", "h3"]):
			if headings.name == 'h1' and headings.text not in serp.keys():
				serp['h1'].append(headings.text.replace('\n',""))
			elif headings.name == 'h2' and headings.text not in serp.keys():
				serp['h2'].append(headings.text.replace('\n',""))
			elif headings.name == 'h3' and headings.text not in serp.keys():
				serp['h3'].append(headings.text.replace('\n',""))

	for url in serp['urls']:

		try:
			response = natural_language_understanding.analyze(
			url=url,
			language="en", 
			features=Features(
				entities=EntitiesOptions(emotion=True, sentiment=True, limit=50),
				keywords=KeywordsOptions(emotion=True, sentiment=True,
										 limit=50))).get_result()
		except (TypeError, SSLError, ApiException) as e:
			st.error(f"Error found! {e} Continuing...")
			continue

		for kw in range(0,len(response["keywords"])):
			cats_kw["Keyword Text"].append(response["keywords"][kw]["text"].lower().replace("`", "'"))
			cats_kw["Keyword Count"].append(response["keywords"][kw]["count"])
			cats_kw["Keyword Relevance"].append(response["keywords"][kw]["relevance"])
			
		for entity in range(0,len(response["entities"])):
			cats_ent["Entity Text"].append(response["entities"][entity]["text"])
			cats_ent["Entity Count"].append(response["entities"][entity]["count"])
			cats_ent["Entity Type"].append(response["entities"][entity]["type"])
			cats_ent["Entity Relevance"].append(response["entities"][entity]["relevance"])			

	def download_link(object_to_download, download_filename, download_link_text):
	    """
	    Generates a link to download the given object_to_download.

	    object_to_download (str, pd.DataFrame):  The object to be downloaded.
	    download_filename (str): filename and extension of file. e.g. mydata.csv, some_txt_output.txt
	    download_link_text (str): Text to display for download link.

	    Examples:
	    download_link(YOUR_DF, 'YOUR_DF.csv', 'Click here to download data!')
	    download_link(YOUR_STRING, 'YOUR_STRING.txt', 'Click here to download your text!')

	    """
	    if isinstance(object_to_download,pd.DataFrame):
	        object_to_download = object_to_download.to_csv(index=False)

	    # some strings <-> bytes conversions necessary here
	    b64 = base64.b64encode(object_to_download.encode()).decode()

	    return f'<a href="data:file/txt;base64,{b64}" download="{download_filename}">{download_link_text}</a>'

	df_kw = pd.DataFrame(cats_kw)
	grouped_df_kw = df_kw.groupby(['Keyword Text']).agg({'Keyword Count': ['sum'], 'Keyword Relevance': ['mean']}).round(3)
	grouped_df_kw = grouped_df_kw.reset_index()
	st.subheader('NLP keyword data')
	st.dataframe(grouped_df_kw)
	grouped_df_kw_csv = grouped_df_kw.to_csv()
	st.download_button(label=f'Download keyword data for {keyword}', data=grouped_df_kw_csv, file_name=f'{keyword}_kw.csv', mime='text/csv')

	df_ent = pd.DataFrame(cats_ent)
	grouped_df_ent = df_ent.groupby(['Entity Text', 'Entity Type']).agg({'Entity Count': ['sum'], 'Entity Relevance': ['mean']}).round(3)
	grouped_df_ent = grouped_df_ent.reset_index()
	st.subheader('NLP entity data')
	st.dataframe(grouped_df_ent)
	grouped_df_ent_csv = grouped_df_ent.to_csv()
	st.download_button(label=f'Download entity data for {keyword}', data=grouped_df_ent_csv, file_name=f'{keyword}_kw.csv', mime='text/csv')

	df = {key:pd.Series(value, dtype='object') for key, value in serp.items()}
	serp_df = pd.DataFrame(df)
	st.subheader('SERP data')
	st.dataframe(serp_df)
	serp_df_csv = serp_df.to_csv()
	st.download_button(label=f'Download SERP data for {keyword}', data=serp_df_csv, file_name=f'{keyword}_kw.csv', mime='text/csv')

	st.markdown('Completed!')
