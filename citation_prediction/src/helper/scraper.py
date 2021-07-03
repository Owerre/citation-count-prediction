####################################
# Author: S. A. Owerre
# Date modified: 09/06/2021
# Class: Web scraping
####################################

# Filter warnings
import warnings
warnings.filterwarnings("ignore")

# Data manipulation
import pandas as pd
import numpy as np

# Request modules
import time
from urllib.request import urlopen
from urllib.error import HTTPError
from bs4 import BeautifulSoup

# Import ADS
import ads

class Scraper:
    """
    A class to scrape and preprocess arXiv papers
    """

    def __init__(self) -> None:
        pass

    def get_data(self, arxiv='physics:cond-mat'):
        """
        This function scrape the condensed matter physics papers,
        and create a pandas dataframe of the data from BeautifulSoup xml parser
        """

        base_url = "http://export.arxiv.org/oai2?verb=ListRecords&"
        url = base_url + 'from=1992-01-01&until=2020-12-31&' + \
            'metadataPrefix=arXiv&set=%s' % arxiv

        df = pd.DataFrame()  # Empty Pandas DataFrame

        while True:

            print("fetching", url)

            try:
                req_xml = urlopen(url)

            except HTTPError as e:
                if e.code == 503:
                    to = int(e.hdrs.get("retry-after", 30))
                    print("Got 503. Retrying after {0:d} seconds.".format(to))
                    time.sleep(to)
                    continue

                else:
                    raise

            b_soup = BeautifulSoup(req_xml, 'xml')
            token = b_soup.resumptionToken

            for record in b_soup.find_all('record'):

                titles = record.find('title').text
                author = [a.text.strip('\n') for a in record.find_all('author')]
                abstracts = record.find('abstract').text
                categories = record.find('categories').text
                arxiv_ids = record.find('id').text
                created = record.find('created').text
                doi = record.find('doi')
                comments = record.find('comments')

                if doi != None:
                    doi = doi.text
                if comments != None:
                    comments = comments.text

                content = {'abstract': abstracts, 'authors': author, 'title': titles, 
                            'categories': categories.split(),'arXiv_id': arxiv_ids, 'date_created': created, 
                            'doi': doi, 'comments': comments
                            }

                df = df.append(content, ignore_index=True)

            if token is None or token.text is None:
                break

            else:
                url = base_url + "resumptionToken=%s" % (token.text)

        return df
    
    def get_cite_ct(self, arXiv_id):
        """
        Get the citation counts of each arXiv paper from ADS (http://adsabs.harvard.edu)
        Note: the ads API allows 5000 requests per day

        Parameter
        --------
        arXiv_id: arxiv ID of the paper

        Return
        -------
        Citation counts of the arxiv ID
        """
        #r = ads.RateLimits('SearchQuery')
        # return r.limits
        q = list(ads.SearchQuery(arXiv=arXiv_id, fl=['id', 'bibcode', 'title', 'citation_count']))
        for paper in q:
            cite_count = paper.citation_count
        return cite_count

    def search_comments(self, comments_string, page_numbers=np.nan, figure_numbers=np.nan):
        """
        Parse through arXiv comments and extract page and figure numbers
        """
        comments = comments_string.split(' ')
        page_number_ind = -1
        figure_number_ind = -1

        for i, com in enumerate(comments):
            if com.find('pages') != -1:
                page_number_ind = i - 1

            if com.find('figures') != -1:
                figure_number_ind = i - 1

        if page_number_ind != -1:
            if comments[page_number_ind].isdigit():
                page_numbers = int(comments[page_number_ind])

        if figure_number_ind != -1:
            if comments[figure_number_ind].isdigit():
                figure_numbers = int(comments[figure_number_ind])

        return [page_numbers, figure_numbers]


