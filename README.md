
# Topsis to find best pretrained model for TEXT GENERATION
This code compares the performance of various pretrained text generation models. It evaluates each model by generating responses to specific prompts and measuring metrics like BLEU and ROUGE. The code then employs TOPSIS analysis to determine the overall best model, considering multiple metrics. 

# Project Structure
1.      Model Selection:
Various text generation models,    including DialoGPT versions, are chosen.
      
2.      Text Generation and Evaluation:


The generated responses are evaluated using BLEU and ROUGE metrics.
Average scores for 
 BLEU,
 ROUGE-1,
 ROUGE-2,
 ROUGE-L,
and response length are calculated.

3.     Results Storage:

Evaluation results are stored in a DataFrame (results_df), saved to 'results.csv'.

4.     TOPSIS Analysis:

TOPSIS is applied to assess the overall performance of models.
The analysis involves normalization, assigning weights and impacts, calculating ideal and worst-ideal values, and computing performance scores.



5.     Visualize TOPSIS Results:

A horizontal bar graph visualizes TOPSIS scores and rankings of different models.








## TOPSIS SCORE
<img src="Topsis score.png" alt="Response Length Comparison">
- ROUGE-1_comparison, ROUGE-2_comparison, ROUGE-L_comparison, Response Length_comparison, Response Length:Bar graphs comparing individual metrics.
<img src="ROUGE-1.png" alt="Response Length Comparison">
<img src="ROUGE-2.png" alt="Response Length Comparison">
<img src="ROUGE-L.png" alt="Response Length Comparison">
<img src="Response length.png" alt="Response Length Comparison">

## Dependencies
- torch: PyTorch library for deep learning.
- transformers: Hugging Face Transformers library for pre-trained language models.
- nltk: Natural Language Toolkit for BLEU score calculation.
- rouge-score: Library for ROUGE score calculation.
- pandas: Data manipulation library.
- matplotlib: Plotting library.
- numpy: Numerical computing library.



