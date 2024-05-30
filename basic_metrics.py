from nltk.sentiment import SentimentIntensityAnalyzer
import textstat
from nltk.tokenize import sent_tokenize, word_tokenize

class metrics:
    def __init__(self, text):
        self.text = text

    def sentence_count(self):  
        sentences = sent_tokenize(self.text)  
        return len(sentences)  
    
    def word_count(self):  
        words = word_tokenize(self.text)  
        return len(words)  
    
    def paragraph_count(self):  
        paragraphs = self.text.split("\n\n")  
        return len(paragraphs)  
    
    def avg_words_per_sentence(self):  
        sentences = sent_tokenize(self.text)  
        words_per_sentence = [len(word_tokenize(sentence)) for sentence in sentences]  
        return sum(words_per_sentence) / len(words_per_sentence)  

    def lexical_diversity(self):  
        words = word_tokenize(self.text)  
        unique_words = set(words)  
        
        if len(words) > 0:  
            lexical_diversity = len(unique_words) / len(words)  
        else:  
            lexical_diversity = 0  
        
        return lexical_diversity

    
    def content_length(self):  
        return len(self.text)  

    def sentiment_score(self):  
        analyzer = SentimentIntensityAnalyzer()  
        sentiment = analyzer.polarity_scores(self.text)  
        compound_score = sentiment['compound']  
        normalized_score = (compound_score + 1) / 2  
        return normalized_score  

    def reading_ease(self):
        return textstat.flesch_reading_ease(self.text)
    
    def keyword_density(self, keyword):
        """
        Keyword Density = ( KR / ( TW -( KR x ( NWK-1 ) ) ) ) x 100

        KR = how many times you repeated a key-phrases

        NWK = number of words in your key-phrases

        TW = total words in the analyzed text
        """  
        self.text = self.text.lower()
        keyword = keyword.lower()
        keyword_count = self.text.count(keyword)  
        keyword_length = len(keyword.split())  
        total_words = len(self.text.split())  
        
        if keyword_length > 0:  
            keyword_density = (keyword_count / (total_words - (keyword_count * (keyword_length - 1)))) * 100  
        else:  
            keyword_density = 0  
        
        return keyword_density  
