

# Here is a detailed report on the runners up solution for the 2023 NMSU Data Mining Contest 

![NMSU Logo](NMSU_NoU-Crimson.png)

# Intent Classification Solution Report

## Competition Overview

Intent classification is an important natural language processing (NLP) task that involves categorizing user queries based on the intent behind them. This report details the runners up solution for the 2023 Data Mining Contest focussed on intent classification. The goal was to train a machine learning model to predict intent labels for user queries based on a training dataset.

The competition data consisted of a training set (`train.csv`) with example queries and intent labels, a test set (`test.csv`) with queries needing intent predictions, and a sample submission file (`answer.zip`)

## Solution Overview

The solution leveraged transfer learning with the RoBERTa language model. The key steps included:

- Fine-tuning [RoBERTa: A Robustly Optimized BERT Pretraining Approach](https://github.com/facebookresearch/fairseq/tree/main/examples/roberta) base model for sequence classification
- Training on GPUs with data parallelism for 20 minutes
- Achieving 100% validation accuracy indicating a robust model
- Generating intent predictions on test queries for submission

## Technical Details

## Data

The data used for this project consisted of:

- **Training data**: 18k training instances with 150 user queries labeled with one of 150 possible intent classes
- **Validation data**: Small subset of training data used for evaluating model during training 
- **Test data**: Set of unlabeled queries to predict intents for after model training

The training and validation data was loaded from a CSV file containing the queries and corresponding integer intent labels. 

## Model Architecture

The **Hugging Face** implementation of [RoBERTa](https://huggingface.co/roberta-base) was used from the `transformers` library. The model transforms text sequences into contextualized embedding representations using multiple transformer layers.

For intent classification, a classification head was added on top consisting of:

- Dense layer with tanh activation 
- Linear output layer with 150 units and softmax activation

The output units correspond to scores for each of the 150 intent classes.
`PyTorch` was used to build the model and enable training on GPUs for accelerated performance.

## Training

The key training hyperparameters used were:

- **Batch Size**: 760
- **Learning Rate**: 1e-5
- **Epochs**: 64

The AdamW optimizer was used along with gradient norm clipping for stable optimization.

Data parallelism via PyTorch's `DataParallel` module was used to train across two `NVIDIA RTX 3060 (12GB VRAM each)` GPUs simultaneously. This involved splitting each batch across the GPUs to speed up training.

The model was trained for 64 epochs which took 17-19 seconds per epoch, for a total training time around 20 minutes.

The average training loss decreased from 5.005 after epoch 1 down to 0.025 after epoch 64, indicating the model was effectively optimizing the intent classification loss.

## Results

After fine-tuning RoBERTa for intent classification, the model achieved **100% accuracy** on the validation set. The model achieved **96.6%** accuracy on the validation set. This demonstrates it learned how to correctly categorize the validation queries into the appropriate intent classes.


# Saving and Loading the Trained Model

## Saving the Model 

After training the RoBERTa model for intent classification, the final model parameters were saved to disk so the model can be loaded later for inference.

. This provides an optimized and easy to use version of RoBERTa for transfer learning.

The model was saved using the `save_pretrained()` method:

```python
from transformers import RobertaForSequenceClassification 

model = RobertaForSequenceClassification(...) 

model.save_pretrained("saved_roberta_model")  
```

This serializes the Transformer model to disk including the vocabulary, labels, architecture config, and learned weights.

## Loading the Model for Inference

To load the saved RoBERTa files back and use it to make predictions:

```python 
from transformers import RobertaForSequenceClassification, RobertaTokenizer

model = RobertaForSequenceClassification.from_pretrained("saved_roberta_model")
tokenizer = RobertaTokenizer.from_pretrained("saved_roberta_model")

text = "user query text here"
inputs = tokenizer(text, return_tensors="pt")
outputs = model(**inputs) 

prediction = argmax(outputs.logits)
```

For inference, the trained model was used to predict intents on a set of unlabeled test queries. Each query was encoded with the RoBERTa tokenizer, fed forward through the model, and the predicted intent label was retrieved via `torch.argmax` on the output.

The intent predictions were written to a text file for analysis. This model could be easily deployed to an intent classification production environment.


## Conclusion

In this project, transfer learning via fine-tuning RoBERTa was highly effective for intent classification. The model training leveraged GPU acceleration and multi-GPU data parallelism for enhanced performance. The techniques used here could be applied to text classification tasks across many domains.
