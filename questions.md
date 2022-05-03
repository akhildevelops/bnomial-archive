## Date - 2022-04-24

## Title - Mia and the feedback loop

### **Question** :

Mia was crushing her thesis!

She was about to release a new neural network architecture that promised to raise the bar on image classification problems.

Mia did not start from scratch. She modified an existing model but added a key ingredient: feedback loops.

A feedback loop is when connections between units form a directed cycle, thus creating loops in the network. This gave Mia's network the ability to save information in the hidden layers.

Mia did a lot of research before deciding in favor of this architecture. She knew the advantages of her decision.

**Which was the architecture that Mia studied to learn about feedback loops?**

### **Choices** :

- Recurrent Neural Networks
- Convolutional Neural Network
- Multilayer Perceptron
- Radial Basis Function Network

---
## Date - 2022-04-25


## Title - Harper and the small gradients


### **Question** :

Harper's team is struggling with the deep neural network they have been building.

Unfortunately, during backpropagation, the gradient values of their network decrease dramatically as the process gets closer to the initial layers, preventing them from learning at the same pace as the last set of layers.

Harper knows their model suffers from the vanishing gradient problem. She decides to research every possible option to improve their model.

**Which of the following techniques will make Harper's model more robust to the vanishing gradient problem?**


### **Choices** :

- Harper should try ReLU as the activation function since it's well-known for mitigating the vanishing gradient problem.
- Harper should modify the model architecture to introduce Batch Normalization.
- Harper should make sure they are initializing the weights properly. For example, using He initialization should help with the vanishing gradient problem.
- Harper should increase the learning rate to avoid getting stuck in local minima and thus reduce the chance of suffering vanishing gradients.

-----------------------

## Date - 2022-04-26


## Title - Exploring data before anything else


### **Question** :

An essential step in any machine learning project is the Exploratory Data Analysis process.

Before we can train a model, we need to understand our data. As the name suggests, Exploratory Data Analysis allows us to explore the data to discover potential problems or patterns that we can use.

**Which of the following are some of the steps we take during this process?**


### **Choices** :

- Learn the distribution of the target variable.
- Understand the features in the dataset and the distribution of their values.
- Evaluate the performance of our models on this data.
- Assess the data quality, including missing or corrupt values.

-----------------------

## Date - 2022-04-27


## Title - Susan needs to make a decision


### **Question** :

The deadline is approaching, and Susan still hasn't decided which version of her classification model to deploy to production.

She experimented with different hyperparameters, and now she has two models that perform pretty well. 

Her problem is that none of these models is better than the other in every situation. One model has a higher recall but worse precision than the other. Susan can improve the precision by playing with different thresholds, but now the recall decreases.

**How can Susan decide which is the best overall model?**


### **Choices** :

- Susan should tune the thresholds until both have a recall of 95% and choose the one with higher precision.
- Susan should tune the thresholds until both have a precision of 95% and choose the one with a higher recall.
- Susan should compute the area under the curve for both models and choose the one with the higher value.
- There's no objective way to decide which model is best. Susan should pick either one of them.

-----------------------

## Date - 2022-04-28


## Title - Linear regression by hand


### **Question** :

The best way to learn something new is to rip the band-aid and tackle a problem from scratch.

Imagine you get a dataset with thousands of samples of houses sold in the U.S. over the last five years. We know the value of a few different features of each home and the price it was sold for. The goal is to build a simple model capable of predicting the price of a new house given those features.

A linear regression model seems like an excellent place to start. 

But you are not writing any code yet. You want to do this manually, starting with a matrix `X` containing the value of the features and a vector `w` containing the weights.

The next step is to multiply `X` and `w`, but you aren't sure about the result of this operation.

**Which of the following better describes the result of multiplying `X` and `w`?**


### **Choices** :

- The result will be a vector `y` containing the actual price of each house as provided in the dataset.
- The result will be a vector `y` containing the predicted price of each house.
- The result will be a matrix `y` containing the actual price of each house and the features from the matrix `X`.
- The result will be a matrix `y` containing the predicted price of each house and the features from the matrix `X`.

-----------------------

## Date - 2022-04-29


## Title - 20,000 sunny and cloudy samples


### **Question** :

Today is your very first day.

You get access to weather data. Twenty thousand samples with the weather of sunny and cloudy days. You want to build a model to predict whether a future day will be sunny or cloudy.

You already know this is a binary classification problem, and now it's time to pick a model.

**Which of the following techniques can you use to build a binary classification model?**


### **Choices** :

- Logistic Regression
- k-Nearest Neighbors
- Neural Networks
- Decision Trees

-----------------------

## Date - 2022-04-30


## Title - The true meaning of hyperparameter tuning


### **Question** :

Marlene is trying to build an audience.

Writing content seems easy, but taking a complex subject and boiling it down to its essence is not an obvious task.

Marlene wants to start from the basics and write as much as possible about the fundamentals of machine learning.

She picked her first topic: hyperparameter tuning.

**If you were trying to summarize the core idea of hyperparameter tuning, which one of the following sentences would you use?**


### **Choices** :

- Hyperparameter tuning is about choosing the set of optimal features from the data to train a model.
- Hyperparameter tuning is about choosing the set of optimal samples from the data to train a model.
- Hyperparameter tuning is about choosing the optimal parameters for a learning algorithm to train a model.
- Hyperparameter tuning is about choosing the set of hypotheses that better fit the goal of the model.

-----------------------

## Date - 2022-05-01


## Title - One of these shouldn't be here


### **Question** :

Here are four different techniques commonly used in machine learning.

Although they are all related somehow, one of them is different from the rest. Your goal is to determine which of the following doesn't belong on this list.

**Can you select the odd one out?**


### **Choices** :

- Expectation–Maximization
- PCA
- DBSCAN
- K-Means

-----------------------

## Date - 2022-05-02


## Title - The bankruptcy story


### **Question** :

Suzanne wants to build an algorithm to predict whether a company is about to declare bankruptcy over the next few months.

She has access to a labeled dataset with detailed financial information from thousands of companies, including those that have declared bankruptcy over the last 100 years.

Suzanne has some ideas but would love to hear what you think.

**Understanding that there are many ways to approach a problem, what would be your first recommendation to Suzanne?**


### **Choices** :

- The best way to approach this problem is with Supervised Learning by using a regression algorithm.
- The best way to approach this problem is with Supervised Learning by using a classification algorithm.
- The best way to approach this problem is with Unsupervised Learning by using a clustering algorithm.
- The best way to approach this problem is with Reinforcement Learning.

-----------------------

## Date - 2022-05-03


## Title - A batch of rotated pictures


### **Question** :

After looking at the last batch of images, the problem was apparent:

Customers were taking pictures and sending them with different degrees of rotation. The Convolutional Neural Network that Jessica built wasn't ready to handle this.

She knew she needed to do something about it.

A couple of meetings later, Jessica knew what the right solution was. It took some time for the team to agree, but they had a plan now.

**Which of the following approaches could Jessica have proposed?**


### **Choices** :

- Extending the pipeline with a data preprocessing step to properly rotate every image coming from the customer before giving the data to the model.
- Extending the model with a layer capable of rotating the data to the correct position.
- Extending the training data with samples of images rotated across the full 360-degree spectrum to build some rotation invariability into the model.
- Configuring the network correctly since Convolutional Neural Networks are translation and rotation invariant and should handle these images correctly.

-----------------------

