from deepeval import evaluate
from deepeval.models import GPTModel
from deepeval.metrics import GEval, HallucinationMetric, FaithfulnessMetric
from deepeval.test_case import LLMTestCaseParams, LLMTestCase
import pandas as pd


class LLM_Metrics:
    def __init__(self, content, reference_text, content_type, llm_model_name='gpt-4o'):
        self.content = content
        self.reference_text = reference_text
        self.content_type = content_type

        # Common LLM metrics defined in common_metrics.csv, computed for ALL types of content
        self.common_metric_details = pd.read_csv('common_metrics.csv')

        metric_df = pd.read_csv('category_metrics.csv')
        metric_df = metric_df[metric_df.Category==self.content_type].drop('Category',axis=1)

        # Category specific metrics. See category_metrics.csv for format details
        self.category_specific_metrics = metric_df

        # combine common and category specific metrics in a single dataframe to be used in compute_metrics
        self.all_metrics = pd.concat([self.common_metric_details, self.category_specific_metrics], ignore_index=True)

        self.deepeval_llm_model = GPTModel(model=llm_model_name, temperature=0)

    @staticmethod
    def parse_metrics(results):
        scores = {}
        metrics = results[0].metrics
        for m in metrics:
            scores.update({m.name: {"score": m.score, "reason": m.reason}})

        return scores

    def get_common_deepeval_metrics(self):
        """These are metrics present in deepeval library. These are optimised for calculating hallucination, faithfullness ,etc score
        While we could create a hallucination score in common_metrics.csv, it would not be as accurate as these metrics
        """
        common_metrics = {}

        # Hallucination metric
        hallucination_test_case = LLMTestCase(
            input='',
            actual_output=self.content,
            context=[self.reference_text]
        )
        hallucination_metric = HallucinationMetric(threshold=0.5, model=self.deepeval_llm_model)
        hallucination_metric.measure(hallucination_test_case)

        common_metrics.update(
            {
                "Hallucination Score": {
                    "score": hallucination_metric.score, 
                    "reason": hallucination_metric.reason,
                }
            }
        )

        fathfullness_test_case = LLMTestCase(
            input='',
            actual_output=self.content,
            retrieval_context=[self.reference_text]
        )
        faithfulness_metric = FaithfulnessMetric(model=self.deepeval_llm_model)
        faithfulness_metric.measure(fathfullness_test_case)
        common_metrics.update(
            {
                "Faithfulness Score": {
                    "score": faithfulness_metric.score,
                    "reason": faithfulness_metric.reason,
                }
            }
        )
        
        return common_metrics

        
        
    def compute_metrics(self):
        metrics_with_reference = []
        content_based_metrics = []

        test_case_with_reference = [LLMTestCase(input=self.reference_text, actual_output=self.content)]
        content_based_test_case = [LLMTestCase(input="",actual_output=self.content)] 

        for idx, row in self.all_metrics.iterrows():
            metric_name = row['metric']
            metric_criteria = row['metric_description'] 

            if row['requires_reference']:
                metric = GEval(
                    name=metric_name,
                    criteria=metric_criteria,
                    model=self.deepeval_llm_model,
                    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT]
                )

                metrics_with_reference.append(metric)
            else:
                metric = GEval(
                    name=metric_name,
                    criteria=metric_criteria,
                    model=self.deepeval_llm_model,
                    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT]
                )

                content_based_metrics.append(metric)

        reference_metric_results = evaluate(test_case_with_reference, metrics_with_reference, use_cache=True, print_results=False)
        content_metric_results = evaluate(content_based_test_case, content_based_metrics, use_cache=True, print_results=False)

        reference_metric_results = LLM_Metrics.parse_metrics(reference_metric_results)
        content_metric_results = LLM_Metrics.parse_metrics(content_metric_results)

        return reference_metric_results, content_metric_results
