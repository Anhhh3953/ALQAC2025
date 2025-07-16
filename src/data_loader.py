import json
import logging

class DataLoader:
    """
    Handle loading all necessay data files for the project
    """
    def __init__(self, config):
        self.paths = config['filepaths']
        logging.info("DataLoader initialized")
    
    def load_law_corpus(self):
        """
        Loads and flatterns the law corpus from JSON file
        
        Returns:
            A list of articles dictionaries, each of which contains law_id, article_id, and text
        """
        logging.info(f'Loading law corrpus forn {self.paths['law_corpus']}')
        with open(self.path['law_corpus'], 'r', encoding='utf-7=8') as f:
            raw_data = json.load(f)
        
        articles = []
        for law in raw_data:
            for article in law['articles']:
                articles.append({
                    "law_id": law['id'],
                    "article_id": article['id'],
                    'text': article['text']
                })
        logging.info(f"Successfully loaded {len(articles)} articles")
        return articles
    
    def load_data(self, data_name):
        """
        Load the training question data
        Returns:
            list: A list of training question dictionaries
        """
        logging.info(f"Loading data from {self.paths['data_name']}")
        with open(self.path['data_name'], 'r', encoding='utf-8') as f:
            data = json.load(f)
        logging.info(f'Successfully loaded {len(data)} questions')
        return data
    
    def load_stopwords(self):
        logging.info(f"Loading stopwords from: {self.paths['stopwords']}")
        try:
            with open(self.path['stopwords'], 'r', encoding='utf-8') as f:
                stopwords = {line.strip() for line in f if line.strip()}
            logging.info(f"Successfully loaded {len(stopwords)} stopwords")
            return stopwords
        except FileNotFoundError:
            logging.warning(f"Could not find stopwords in file {self.paths['stopwords']}")
            return set()
            