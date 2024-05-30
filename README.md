# content_evaluator
LLM powered content evaluator. Score Blogs and Emails on marketing specific metrics 

## Usage

```python
import os
os.environ['OPENAI_API_KEY'] = 'your-openai-key-here'

from base_content import Content

reference_details = "Product: VisionX Immersion VR Headset Display: Dual 4K OLED FOV: 200° Tracking: 6DOF inside-out Connectivity: WiFi 6, Bluetooth 5.2, USB-C"
blog = """Exciting news for VR enthusiasts! We're thrilled to introduce the VisionX Immersion VR Headset, redefining your virtual reality experience. With dual 4K OLED display panels, this headset promises crystal-clear visuals. A standout feature is its 200° FOV and advanced 6DOF inside-out tracking system, bringing your VR world to life like never before. To top it off, it supports WiFi 6, Bluetooth 5.2, and USB-C connectivity, promising a seamless and immersive VR experience. Stay tuned for more updates!"""

content = Content(generated_content=blog,
                  reference_text=reference_details,
                  content_type='blog',
                  model='gpt-4o',
                  target_keywords = 'VisionX latest VR')

content.get_statistical_metrics()
content.get_content_insights()
content.get_llm_metrics()
```

## [base_content.py](base_content.py)
Content class takes in the following params:
- generated_content - The content to be evaluated
- reference_text - the source information based on which the content is generated
- content_type - blog, email, tweet, etc
- kwargs - Additional keyword arguments which might be relevant. Brand tone, Targeted Keywords

(Note: Ensure that keyword arguments passed have exaplainable names. Also some metrics like [Keyword Usage](category_metrics.csv#L3), need keyword arguments like target_keywords to be passed when creating the Content object. Absence of keyword arguments required in metrics may affect evaluation score)

## [llm_metrics.py](llm_metrics.py)
Uses [deepeval](https://docs.confident-ai.com/docs/getting-started) to compute various metrics using LLM. Some are pre-defined metrics like Hallucination and Faithfulness, while others are custom defined metrics which are emplemented with [G-Eval](https://docs.confident-ai.com/docs/metrics-llm-evals). G-Eval helps to create metrics using custom criteria defined in [common_metrics.csv](common_metrics.csv) and [category_metrics.csv](category_metrics.csv). 

## [common_metrics.csv](common_metrics.csv)
Common metrics which can be used to evaluate ANY type of content. These metrics will be evaluated for ALL types of content, so the criteria for evaluation should be generalisable across categories.

To add more common metrics, these 3 details need to be added in the csv: 
1. metric name
2. description of what it measures
3. whether this metric requires reference material to evaluate or not (for eg. to get [factual coverage](common_metrics.csv#L3) score, the content needs to be compared to the core facts/details, so this variable will be true)  

## [category_metrics.csv](category_metrics.csv)
Metrics to evaluate content belong to a specific category. Whether the content is optimised for mobile would be a metric for blogs since mobile optimization is a major factor for SEO, while a catchy subject line would be a more useful metric for outreach emails. 

To add a new metric, the process is similar to common metrics, along with an additional step of adding the category also

## [llm_helper.py](llm_helper.py)
Helper class to get content insights like tone, central topic, keywords. Can be modified to include additional details


## [basic_metrics.py](basic_metrics.py)
Helper class to get statistical metrics like word count, reading ease, length and more

# Extending the evaluation to new categories and metrics

Current functionality implements metrics for email and blog content. Now to add support for another type of content, let's say tweets, the following process needs to be done:
1. Identify some metrics which can be used to evaluate tweets objectively (presence of hashtags, relevance to current trending topics, etc)
2. Create metrics out of this and add to category_metrics.csv with appropriate description and booleans for whether these metrics require reference text or not
3. Add tweet in Category column for all these metrics

Now you can initialise a Content object with content type as tweet and get metrics to score/evaluate the tweet content
