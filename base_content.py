from basic_metrics import metrics
from llm_metrics import LLM_Metrics
from llm_helper import insights_chain

class Content:
    def __init__(self, generated_content, reference_text, content_type, model='gpt-4o', **kwargs):
        self.content = generated_content
        self.reference_text = reference_text
        self.content_type = content_type
        
        if kwargs:
            self.reference_text = 'Core Information : \n' + reference_text + "\n___________________"
            print('INFO : Additional Keywrod arguments passed. ALL keyword arguments will be appended to reference text')
            print('Avoid irrelevant keyword arguments. This will ensure more stable evaluation')
            for variable, value in kwargs.items():
                self.reference_text += "\n " + variable + " : \n " + value + "\n___________________"
        
        
        self.basic_metrics = metrics(self.content)
        self.llm_metrics = LLM_Metrics(content=self.content, reference_text=self.reference_text, content_type=self.content_type, llm_model_name=model)

        self.insights = insights_chain
        
        
    def get_statistical_metrics(self):
        """Common metrics which will be present across every content category"""
        
        overall_sentiment = self.basic_metrics.sentiment_score()
        reading_ease = self.basic_metrics.reading_ease()
        word_count = self.basic_metrics.word_count()
        para_count = self.basic_metrics.paragraph_count()
        sent_count = self.basic_metrics.sentence_count()
        words_per_sentence = self.basic_metrics.avg_words_per_sentence()
        lexical_diversity = self.basic_metrics.lexical_diversity()
        
        stat_metrics = locals()
        
        return {**stat_metrics}

    def get_llm_metrics(self):
        all_llm_metrics = {}
        
        common_deepeval_metrics = self.llm_metrics.get_common_deepeval_metrics()
        reference_based_metrics, content_based_metrics = self.llm_metrics.compute_metrics()
        
        all_llm_metrics.update(common_deepeval_metrics)
        all_llm_metrics.update(reference_based_metrics)
        all_llm_metrics.update(content_based_metrics)
        
        return all_llm_metrics
    
    def get_content_insights(self):
        content_insights = self.insights.invoke({'category':self.content_type,'content':self.content})
        
        return content_insights 
        

