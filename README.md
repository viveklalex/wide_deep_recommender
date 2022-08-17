
# Wide & Deep Recommender
[![MIT License](https://img.shields.io/apm/l/atomic-design-ui.svg?)](https://github.com/tterb/atomic-design-ui/blob/master/LICENSEs)

Wide & Deep recomendation system using tensorflow
## Introduction 

 
![alt text](https://raw.githubusercontent.com/vivekalex61/resume_ner/main/images/)
      
#### Recomendation systems?

A recommender system can be viewed as a search ranking
system, where the input query is a set of user and contextual
information, and the output is a ranked list of items. Given a query, the recommendation task is to find the relevant
items in a database and then rank the items based on certain
objectives, such as clicks or purchases.


#### Two principal approaches to recommender systems.
- The first is the **content-based approach**, which makes use of features for both users and items. Users may be described by properties such as age and gender, and items may be described by properties such as author and manufacturer. Typical examples of content-based recommendation systems can be found on social matchmaking sites.
- The second approach is **collaborative filtering**, which uses only identifiers of the users and the items and obtains implicit information about these entities from a (sparse) matrix of ratings given by the users to the items. We can learn about a user from the items they have rated and from other users who have rated the same items.
#### Wide and Deep approach

The Wide & Deep recommender combines these approaches, using collaborative filtering with a content-based approach. It is therefore considered a **hybrid recommender**.

One Challenge in recommender systems, similar to the general search ranking problem, is to achieve both memorization
and generalization.


#### What are memorization and generalization ?

#### Memorization
Memorisation can be loosely defined as learning the frequent co-occurrence of items or features and exploiting the correlation available in the historical data.

1. Memorisation of feature interactions through a wide set of cross-product feature transformations are effective and interpretable, while generalisation requires more feature engineering effort.

2. Recommendations based on memorisation are usually more topical and directly relevant to the items on which users have already performed actions.

3. Memorisation can be achieved effectively using cross-product transformations over sparse features. This explains how the co-occurrence of a feature pair correlates with the target label.

4. One limitation of cross-product transformations is that they do not generalise to query-item feature pairs that have not appeared in the training data.

5. Wide linear models can effectively memorise sparse feature interactions using cross-product feature transformations.



#### Generalisation
Generalisation, on the other hand, is based on transitivity of correlation and explores new feature combinations that have never or rarely occurred in the past.

1. With less feature engineering, deep neural networks can generalise better to unseen feature combinations through low-dimensional dense embeddings learned for the sparse features.

2. However, deep neural networks with embeddings can over-generalise and recommend less relevant items when the user-item interactions are sparse and high-rank.

3. Generalisation tends to improve the diversity of the recommended items. Generalisation can be added by using features that are less granular , but manual feature engineering is often required.

4. For massive-scale online recommendation and ranking systems in an industrial setting, generalised linear models such as logistic regression are widely used because they are simple, scalable and interpretable. The models are often trained on binarised sparse features with one-hot encoding.
          

#### How Wide & Deep achieve memorization and generalization ?

- **The Wide component**

The wide model, often referred to in the literature as the linear model, memorizes users and their past product choices. Its inputs may consist simply of a user identifier and a product identifier, though other attributes relevant to the pattern (such as time of day) may also be incorporated.

![alt text](https://raw.githubusercontent.com/vivekalex61/resume_ner/main/images)

- **The Deep component**

The deep portion of the model, so named as it is a deep neural network, examines the generalizable attributes of a user and their product choices. From these, the model learns the broader characteristics that tend to favor users’ product selections.

- **Training wide and Deep jointly**

The wide component and deep component are combined
using a weighted sum of their output log odds as the prediction, which is then fed to one common logistic loss function for joint training. 

![alt text](https://raw.githubusercontent.com/vivekalex61/resume_ner/main/images)


## Implementation

- Data generation
- Data Preprocessing
- Exploratory Data Analysis
- Model training
- Model serving

### Data generation

Dataset is collected from  https://www.kaggle.com/datasets/wenruliu/adult-income-dataset

An individual’s annual income results from various factors. Intuitively, it is influenced by the individual’s education level, age, gender, occupation, and etc.

Data consist of 48842 instances, mix of continuous and discrete (train=32561, test=16281)
Columns :



![alt text](https://raw.githubusercontent.com/vivekalex61/resume_ner/main/images)



### Data Preprocessing

We haven't done any preprocessing. The dataset is already cleaned.
Income is used as label. The same step can be followed for the recommendation dataset
### Exploratory Data Analysis

Exploratory Data Analysis using Pandas and visualized using Matploilib

### Model training

During training, our input layer takes in training data and vocabularies and generate sparse and dense features together with a label. The wide component consists of the cross-product transformation of user installed apps and impression apps. For the deep part of the model, a 32-dimensional embedding vector is learned for each categorical feature. We concatenate all the embeddings together with the dense features, resulting in a dense vector of approximately 1200 dimensions. The concatenated vector is then fed into 3 ReLU layers, and finally the logistic output unit.


![alt text](https://raw.githubusercontent.com/vivekalex61/resume_ner/main/images)

### Model serving

Once the model is trained and verified, we load it into the model servers. For each request, the servers receive a set of app candidates from the app retrieval system and user features to score each app. Then, the apps are ranked from the highest scores to the lowest, and we show the apps to the users in this order. The scores are calculated by running a forward inference pass over the Wide & Deep model.


Each candidate receives a probability score from the logistic output unit .
y∈[0,1].
## Results

Evaluation Metrics :

Metrics:

- Sensitivity: The probability that the model predicts a positive outcome for an observation when indeed the outcome is positive. This is also called the **“true positive rate.”**
- Specificity: The probability that the model predicts a negative outcome for an observation when indeed the outcome is negative. This is also called the **“true negative rate.”**
One way to visualize these two metrics is
 by creating a **ROC** curve, which stands for “receiver operating characteristic” curve.

This is a plot that displays the sensitivity along the y-axis and (1 – specificity) along the x-axis.

One way to quantify how well the model classifying data is to 
calculate **AUC**, which stands for “area under curve.”

The value for AUC ranges from 0 to 1. A model that has an AUC of 1 is able to perfectly classify observations into classes while a model that has an AUC of 0.5 does no better than a model that performs random guessing.
 
Here ,

AUC Score : .67

- 0.5 = No discrimination
- 0.5-0.7 = Poor discrimination
- 0.7-0.8 = Acceptable discrimination
- 0.8-0.9= Excellent discrimination
- >0.9 = Outstanding discrimination


Classification Report 

![alt text](https://raw.githubusercontent.com/vivekalex61/resume_ner/main/images/)



## Contributing

Contributions are always welcome!

## Reference

https://www.databricks.com/blog/2021/06/09/how-to-build-a-scalable-wide-and-deep-product-recommender.html

https://docs.microsoft.com/en-us/azure/machine-learning/component-reference/train-wide-and-deep-recommender

https://www.statology.org/what-is-a-good-auc-score/

https://calvinfeng.gitbook.io/machine-learning-notebook/supervised-learning/recommender/wide_and_deep_learning_for_recommender_systems