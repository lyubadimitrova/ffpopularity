import spacy
from textstat.textstat import textstat
from textblob import TextBlob
from pathlib import Path 
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer
import sys
import argparse
import pickle


class FeatureExtractor():

	def __init__(self, input_dir=''):
		self.input_dir = Path(input_dir)
		self.df = None
		self.first_feature_idx = None
		self.tags2_features = None

	
	def read_aggregate(self):
		"""
		Reads the fanwork metadata csv files and aggregates them to a single DataFrame.
		Some cleaning happens, too.
		"""

		cols = ['work_id', 'title', 'rating', 'category', 'language',
	       'fandom', 'relationship', 'character', 'additional tags', 'published',
	       'words', 'hits', 'summary']

		aux_dfs = []
		for csv_file in self.input_dir.iterdir():
			aux_df = pd.read_csv(csv_file, usecols=cols)
			aux_df = aux_df[aux_df['language']=='English']    # removes shifted columns
			aux_df = aux_df[~aux_df['words'].isnull()]		  # no idea why some works have null (not 0!) words
			aux_df = aux_df[~aux_df['hits'].isnull()]		  # removes works with hidden hit counts
			aux_df = aux_df[~aux_df['work_id'].isnull()]
			aux_df = aux_df[~aux_df['title'].isnull()]
			aux_dfs.append(aux_df)

		df = pd.concat(aux_dfs, ignore_index=True)

		df = df[~df.duplicated('work_id')]      # since the scraping wasn't done at the same time every day, some works may have been scraped
											    # during two consecutive days.
		
		df[['work_id', 'words', 'hits']] = df[['work_id', 'words', 'hits']].astype(int)
		df['published'] = pd.to_datetime(df.published)

		df = df.loc[(df.published >= '2018-07-30') & (df.published <= '2018-09-30')]    # there were some posting dates from 2012?
		df = df.fillna('')

		print('Initial DataFrame done')
		self.df = df


	def prep(self):
		"""
		Prepares the DataFrame for the feature extraction. Takes a while. 
		If DF filtering is necessary, it should be done BEFORE the prepping, to save time.
		"""
		
		df = self.df

		replacements = {('“', '"'), ('”', '"')}
		for repl in replacements:
			df['summary'] = df.summary.str.replace(repl[0], repl[1])

		nlp = spacy.load("en")

		df['cleaned'] = df.summary.apply(lambda x: nlp(x))
		print('cleaned done')
		df['cl_title'] = df.title.apply(lambda x: nlp(x, disable=['parser', 'ner']))
		print('cl_title done')
		df['tags_list'] = df['additional tags'].apply(lambda x: x.split(', ') if isinstance(x, str) else '')
		print('tags_list done')
		df['cl_tags'] = df.tags_list.apply(lambda x: [nlp(tag, disable=['parser', 'tagger', 'ner']) for tag in x])
		print('cl_tags done')
		df['rels_list'] = df.relationship.apply(lambda x: x.split(', ') if isinstance(x, str) else [])
		print('rels_list done')
		df['chars_list'] = df.character.apply(lambda x: x.split(', ') if isinstance(x, str) else [])
		print('chars_list done')

		print('Prepping done')

		self.first_feature_idx = df.columns.get_loc('chars_list') + 1       # needed for the pickling of the features


	def add_summary_features(self):

		df = self.df

		df['len_tokens'] = df.cleaned.apply(lambda x: len(x))
		df['len_tokens_no_punct'] = df.cleaned.apply(lambda x: len([t for t in x if t.pos_ != 'PUNCT']))
		df['len_sentences'] = df.cleaned.apply(lambda x: len(list(x.sents)))
		df['type_token_ratio'] = df.apply(lambda row: len({token.lower_ for token in row['cleaned']})
			/ row['len_tokens'] if row['len_tokens'] else 0, axis=1)
		# -> primitive spell checking
		df['out_of_vocab'] = df.apply(lambda row: len([t for t in row['cleaned'] if t.lemma_.lower() not in t.vocab and t.pos_ != 'PROPN'])
			/ row['len_tokens_no_punct'] if row["len_tokens_no_punct"] else 0, axis=1)
		df['stopwords'] = df.apply(lambda row: len([t for t in row['cleaned'] if t.is_stop])
			/ row['len_tokens_no_punct'] if row['len_tokens_no_punct'] else 0, axis=1)
		df['punctuation'] = df.apply(lambda row: len([t for t in row['cleaned'] if t.pos_ == 'PUNCT'])
			/ row['len_tokens'] if row['len_tokens'] else 0, axis=1)
		df['quotes'] = df.summary.apply(lambda x: 1 if x.count('"') else 0)      # usually means the summary is an excerpt from the work (there is direct speech)
		df['brackets'] = df.summary.apply(lambda x: 1 if x.count('(') and x.count(")") else 0)
		df['entities'] = df.apply(lambda row: len(row["cleaned"].ents), axis=1)
		df['avg_word_len'] = df.apply(lambda row: sum([len(t) for t in row['cleaned'] if t.pos_ != 'PUNCT'])
			/ row['len_tokens_no_punct'] if row["len_tokens_no_punct"] else 0, axis=1)
		df['avg_sent_len'] = df.apply(lambda row: sum([len(sent) for sent in row['cleaned'].sents])
			/ row['len_sentences'] if row["len_sentences"] else 0, axis=1)
		df['auto_readability_idx'] = df.summary.apply(lambda x: textstat.automated_readability_index(x) if x else -20)
		df['dale-chall_score'] = df.summary.apply(lambda x: textstat.dale_chall_readability_score(x) if x else 0)
		df['adj_ratio'] = df.apply(lambda row: len([t for t in row['cleaned'] if t.pos_ == 'ADJ'])
			/ row['len_tokens_no_punct'] if row["len_tokens_no_punct"] else 0, axis=1)
		df['adv_ratio'] = df.apply(lambda row: len([t for t in row['cleaned'] if t.pos_ == 'ADV'])
			/ row['len_tokens_no_punct'] if row["len_tokens_no_punct"] else 0, axis=1)
		df['verb_ratio'] = df.apply(lambda row: len([t for t in row['cleaned'] if t.pos_ == 'VERB'])
			/ row['len_tokens_no_punct'] if row["len_tokens_no_punct"] else 0, axis=1)
		df['is_ascii'] = df.summary.apply(lambda x: 1 if self.isascii(x) else 0)
		sentiment = df.apply(lambda x: pd.Series(list(TextBlob(x['summary']).sentiment), index=['polarity', 'subjectivity']), axis=1)
		df['polarity'] = sentiment['polarity']
		df['subjectivity'] = sentiment['subjectivity']


	def add_title_features(self):

		df = self.df

		df['title_len'] = df.cl_title.apply(lambda x: len(x))
		df['title_len_char'] = df.cl_title.apply(lambda x: sum([len(t) for t in x]))
		df['title_case_ratio'] = df.apply(lambda row: len([t for t in row['cl_title'] if t.is_title]) / row['title_len'], axis=1)
		df['lowercase_ratio'] = df.apply(lambda row: len([t for t in row['cl_title'] if t.is_lower]) / row['title_len'], axis=1)
		df['uppercase_ratio'] = df.apply(lambda row: len([t for t in row['cl_title'] if t.is_upper]) / row['title_len'], axis=1)
		df['digit_ratio'] = df.apply(lambda row: len([t for t in row['cl_title'] if t.is_digit]) / row['title_len'], axis=1)
		df['punct_ratio'] = df.apply(lambda row: len([t for t in row['cl_title'] if t.is_punct]) / row['title_len'], axis=1)
		df['title_oov_ratio'] = df.apply(lambda row: len([t for t in row['cl_title'] if t.lemma_.lower() not in t.vocab and t.pos_ != 'PROPN'])
			/ row['title_len'], axis=1)
		df['title_is_ascii'] = df.title.apply(lambda x: 1 if self.isascii(x) else 0)
		df['title_stop_ratio'] = df.apply(lambda row: len([t for t in row['cl_title'] if t.is_stop])
			/ row['title_len'], axis=1)
		df['title_verb_ratio'] = df.apply(lambda row: len([t for t in row['cl_title'] if t.pos_ == 'VERB'])
			/ row['title_len'], axis=1)
		df['title_avg_word_len'] = df.apply(lambda row: sum([len(t) for t in row['cl_title']])
			/ row['title_len'], axis=1)


	def add_rating_features(self):
		rating_dict = {'Not Rated':0, 'General Audiences':1, 'Teen And Up Audiences':2, 'Mature':3, 'Explicit':4}
		df = self.df
		df['rating_idx'] = df.rating.apply(lambda x: rating_dict[x])


	def add_relationship_features(self):
		df = self.df
		df['rels_count'] = df.rels_list.apply(lambda x: len(x))
		df['reader_insert'] = df.relationship.apply(lambda x: 1 if isinstance(x, str) and ('Reader' in x) else 0)


	def add_character_features(self):
		df = self.df
		df['character_count'] = df.chars_list.apply(lambda x: len(x))
		df['ratio_originals'] = df.apply(lambda row: len([char for char in row.chars_list if ('Original' in char)]) 
			/ row.character_count if row.character_count else 0, axis=1)


	def add_tags1_features(self):
		def is_canon(tag):   # Canon Tags have capitalized content words
			return all([t.is_title for t in tag if not (t.is_stop or (t.text in '-/():&') or t.is_digit)])

		df = self.df
		df['tags_count'] = df.tags_list.apply(lambda x: len(x))
		df['ratio_canon'] = df.apply(lambda row: len([tag for tag in row['cl_tags'] if is_canon(tag)]) 
			/ row['tags_count'] if row['tags_count'] else 0, axis=1)
		df['ratio_ascii'] = df.tags_list.apply(lambda x: len([self.isascii(tag) for tag in x]) / len(x) if x else 0)
		df['min_tag_len'] = df.cl_tags.apply(lambda x: min([len(tag) for tag in x]) if x else 0)   # length in tokens
		df['max_tag_len'] = df.cl_tags.apply(lambda x: max([len(tag) for tag in x]) if x else 0)
		df['avg_tag_len'] = df.apply(lambda row: sum([len(tag) for tag in row['cl_tags']]) 
			/ row['tags_count'] if row['tags_count'] else 0, axis=1)


	def add_tags2_features(self):
		"""
		Extracts the BOW features for the tags. Saved separately.
		"""
		df = self.df

		df['additional tags'] = df['additional tags'].apply(lambda x: x.replace(', ', ','))

		v = CountVectorizer(analyzer='word', token_pattern="[^,]+", max_features=300)
		self.tags2_features = v.fit_transform(df['additional tags']).toarray()


	def add_category_features(self):
		categories_dict = {'F/F':1, 'F/M':2, 'M/M':3, 'Multi':4, 'Gen':5, 'Other':6}
		df = self.df
		df['int_category'] = df.category.apply(lambda x: categories_dict.get(x, categories_dict['Multi'] if isinstance(x, str) else 0))

	
	def add_length_features(self):
		df = self.df
		df['length'] = df.words


	def extract(self, enable=[], disable=[]):
		"""
		Governs the feature extraction through the enable/disable parameters.

		Args:
		enable - list - Only the features in enable are extracted.
		disable - list - All features but these are extracted. Ignored if enable is not empty. 
		"""
		if enable:
			for attr in enable:
				feature_func = getattr(self, 'add_{}_features'.format(attr))
				feature_func()
				print("{} features added".format(attr.capitalize()))
		
		else:
			feature_funcs = [getattr(self, f) for f in dir(FeatureExtractor) if f.startswith('add')]
			dis = [getattr(self, 'add_{}_features'.format(attr)) for attr in disable]
			for func in feature_funcs:
				if func not in dis:
					func()
			disabled = 'except for ' + ', '.join(disable) if disable else ''
			print("All features added {}".format(disabled))



	def filter_df(self, **filters):
	    """
	    Returns a filtered DataFrame. Possible filters include 
	    fandom, relationship, character, words, hits, category, rating.

	    Args:
	    df - pandas.DataFrame - A DataFrame with fanfic metadata.
	    filters - **kwargs - The filters.
	    """
	    condition = True
	    df = self.df

	    for f in filters:
	        value = filters[f]
	        c = False
	        
	        if f == 'fandom':
	            if isinstance(value, (set, list)):
	                for v in value:
	                    c = c | df[f].str.contains(v, na=False, regex=False)
	            elif isinstance(value, str):
	                c = c | df[f].str.contains(value, na=False, regex=False)
	        
	        elif f in {'words', 'hits'}:
	            if isinstance(value, range):
	                c = c | df[f].isin(value)
	            elif isinstance(value, int):
	                c = c | (df[f] == value)
	                
	        elif f in {'category', 'rating'}:
	            c = c | (df[f] == value)
	            
	        condition = condition & c
	            
	    self.df = df[condition]


	def normalize(self):
		"""
		Normalizes the features. According to the sklearn docs, 
		the RobustScaler handles outliers better than the StandardScaler.
		"""
		features = self.df.iloc[:, self.first_feature_idx:].copy()
		rs = preprocessing.RobustScaler()

		scaled = rs.fit_transform(features)
		self.df.iloc[:, self.first_feature_idx:] = scaled


	@staticmethod
	def isascii(x):
		return len(x) == len(x.encode())


	def save_features(self, output_dir, filename):
		"""
		Pickles the feature vectors and the corresponding labels and work IDs.

		Args:
		output_dir - str - The path where the pickled files should be saved.
		filename - str - Name your project.
		"""

		output_dir = Path(output_dir)
		output_dir.mkdir(parents=True, exist_ok=True)

		vectors = self.df.iloc[:, self.first_feature_idx:].values
		labels = self.df.loc[:, 'hits'].values
		work_ids = self.df.loc[:, 'work_id'].values

		with open(str(output_dir / (filename + '_vectors.p')), 'wb') as f:
			pickle.dump(vectors, f)

		with open(str(output_dir / (filename + '_labels.p')), 'wb') as f:
			pickle.dump(labels, f)

		with open(str(output_dir / (filename + '_bow_tags.p')), 'wb') as f:
			pickle.dump(self.tags2_features, f)

		with open(str(output_dir / (filename + '_ids.p')), 'wb') as f:
			pickle.dump(work_ids, f)

	
