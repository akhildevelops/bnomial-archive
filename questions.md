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

## Date - 2022-05-04


## Title - Alex's model is not doing well


### **Question** :

Alex is a Machine Learning Engineer working for a new photo-sharing startup.

His team started building a model to predict the likeability of every new image posted on the platform. They collected some data and built a simple classification model.

Unfortunately, Alex quickly realizes that the model doesn't perform well. He notices that the training error is not as low as expected.

**What do you think is happening with Alex's model?**


### **Choices** :

- It's very likely that the model suffers from high bias and is underfitting. This usually happens when the model is not complex enough and can't capture the relationship between input and output variables.
- It's very likely that the model suffers from low bias and is underfitting. This usually happens when the model is not complex enough and can't capture the relationship between input and output variables.
- It's very likely that the model suffers from high variance and is overfitting. This usually happens when the model is too complex and captures the noise of the data.
- It's very likely that the model suffers from low variance and is overfitting. This usually happens when the model is too complex and captures the noise of the data.

-----------------------

## Date - 2022-05-05


## Title - Behind Gradient Descent


### **Question** :

It's 2030, and neural networks are taught at high schools worldwide.

It makes sense. Few subjects are as impactful to society as machine learning, so it's only appropriate that schools get students onboard from a very early age.

Lillian spent a long time learning about gradient descent and how it's an optimization algorithm frequently used in machine learning applications.

This is Lillian's last exam. The first question asks her to describe in a few words how gradient descent works.

**Which of the following statements is a sensible description of how the algorithm works?**


### **Choices** :

- Gradient descent identifies the minimum loss and adjusts every parameter proportionally to this loss.
- Gradient descent searches every possible combination of parameters to find the optimal loss.
- Gradient descent identifies the slope in all directions and adjusts the parameters to move them in the direction of the negative slope.
- Gradient descent identifies the slope in all directions and adjusts the parameters to move them in the direction of the slope.

-----------------------

## Date - 2022-05-06


## Title - A recommendation for Adrienne


### **Question** :

Kaggle looked like the perfect opportunity for Adrienne to start practicing machine learning.

She went online and started listening to the conversations about popular Kagglers. One particular topic caught her attention: They kept discussing different ways to create ensembles.

Adrienne knew that ensemble learning is a powerful technique where you combine the decisions from multiple models to improve the overall performance. She had never used ensembles before, so she decided this was the place to start.

**Which of the following are valid ensemble techniques that Adrienne could study?**


### **Choices** :

- Max Voting: Multiple models make predictions for each sample. The final prediction is the one produced by the majority of the models.
- Weighted Voting: Multiple models make predictions for each sample, and each model is assigned a different weight. The final prediction considers the importance of the model in determining the final vote.
- Simple Averaging: Multiple models make predictions for each sample. The final prediction is the average of all of those predictions.
- Weighted Averaging: Multiple models make predictions for each sample, and each model is assigned a different weight. The final prediction is the average of all of those predictions, considering the importance of each model.

-----------------------

## Date - 2022-05-07


## Title - Sometimes, small is better


### **Question** :

Fynn is new to a team working on a neural network model. Unfortunately, they haven't been happy with the results so far.

Fynn thinks that he found the problem: they chose a batch size as large as it fits into the GPU memory. His colleagues believe this is the right approach, but Fynn believes a smaller batch size will be better.

**What would be good arguments to support Fynn's suspicion?**


### **Choices** :

- A smaller batch size is more computationally effective.
- A smaller batch size reduces overfitting because it increases the noise in the training process.
- A smaller batch size reduces overfitting because it decreases the noise in the training process.
- A smaller batch size can improve the generalization of the model.

-----------------------

## Date - 2022-05-08


## Title - Reese's baseline


### **Question** :

Starting with a simple baseline is a great way to approach a new problem.

Reese knew that, and her go-to has always been a simple Linear Regression, probably one of the most popular algorithms in statistics and machine learning.

But Reese knows that for Linear Regression to work, she must consider several assumptions about the problem.

**Which of the following are some of the assumptions that Reese should make for Linear Regression to be a good candidate for her baseline?**


### **Choices** :

- The relationship between the features in the data and the target variable must be linear.
- The features in the data are highly correlated between them.
- The features in the data and the target variable are not noisy.
- There must not be more than two relevant features plus the target variable.

-----------------------

## Date - 2022-05-09


## Title - Migrating to PyTorch Lighting


### **Question** :

Many of the old team members have left Layla's company, forcing them to start building a new team.

They have been hiring from local universities, and most new hires brought a lot of experience in PyTorch Lightning. Unfortunately for Layla's company, their main product uses TensorFlow.

After some discussions, Layla's team decided to migrate their model to PyTorch Lightning. This change, however, will not come without making some concessions. 

**Which of the following are some of the downsides of this decision?**


### **Choices** :

- The team will lose the ability to deploy the model in TPUs (Tensor Processing Units), limiting them to GPUs and CPUs.
- The team won't be able to use tools like TensorBoard during the training process, so they will need to find an equivalent tool compatible with PyTorch Lightning.
- The team will have to invest time to migrate the deployment process of their model from TensorFlow Serving to something like TorchServe or PyTorch Live.
- Migrating the existing codebase to PyTorch Lightning could introduce unforeseen problems that could cause issues with the new model.

-----------------------

## Date - 2022-05-10


## Title - The brains behind transformers


### **Question** :

It took some time, but Kinsley finished replacing her old model based on a [Long Short-Term Memory](https://machinelearningmastery.com/gentle-introduction-long-short-term-memory-networks-experts/) (LSTM) network with a new version using Transformers.

The results of the new model were impressive. The whole team was thrilled with Kinsley's work, and the company organized an internal session for Kinsley to bring everyone up to speed.

After finishing her speech, a coworker asked Kinsley a question: 

**Which company invented the Transformer architecture?**


### **Choices** :

- Hugging Face
- OpenAI
- Google
- Allen Institute of AI


### **Answer** :

<details><summary>CLICK ME</summary><p>0010</p></details>


### **Explaination** :

<details><summary>CLICK ME</summary><p>In 2017, a team at [Google Brain](https://en.wikipedia.org/wiki/Google_Brain) published the now-famous paper ["Attention Is All You Need,"](https://arxiv.org/abs/1706.03762) where they introduced the Transformer architecture, which transforms one sequence into another with the help of an Encoder and a Decoder. 

[Hugging Face](https://huggingface.co) is an AI community that hosts many NLP models, including a large number of transformer models. Despite being a pioneer in adopting transformer models, Hugging Face is not behind the creation of Transformers.

OpenAI is another powerhouse that conducts AI research to promote and develop AI to benefit humanity. OpenAI is behind models like [GPT-3](https://en.wikipedia.org/wiki/GPT-3), [CLIP](https://openai.com/blog/clip/), and [DALL-E](https://openai.com/blog/dall-e/), all of which use the Transformer architecture. OpenAI, however, didn't invent Transformers.

Finally, the [Allen Institute of AI](https://allenai.org) (also known as AI2) is the AI research institute behind [Macaw](https://macaw.apps.allenai.org), a high-performance question-answering model capable of giving GPT-3 a run for its money. Despite their work with Transformers, they aren't behind its creation either.

In summary, the correct answer to this question is that Google is the inventor of Transformers.</p></details>


### **References**: 

<details><summary>CLICK ME</summary><p>* [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
* [GPT-3](https://en.wikipedia.org/wiki/GPT-3)
* [CLIP](https://openai.com/blog/clip/)
* [DALL-E](https://openai.com/blog/dall-e/)
* [Macaw](https://macaw.apps.allenai.org)</p></details>

-----------------------

## Date - 2022-05-11


## Title - Trending recession


### **Question** :

The company's accounting team used a spreadsheet with some rudimentary charts, but it was time to get serious.

That's when Peyton came in.

Payton had a lot of experience doing time series analysis. Her mandate was simple: using the financial data of past years, predict where the company is going over the next few quarters.

Despite Payton's credentials, the team was worried: the company has been slowly recovering from a recession, and they were concerned this would skew future data. 

Payton went over the different components of time series analysis and explained how to classify this specific trend.

**Which of the following was Payton's explanation about this recession period?**


### **Choices** :

- The company's recession is part of a secular variation.
- The company's recession is part of a seasonal variation.
- The company's recession is part of a cyclical variation.
- The company's recession is part of an irregular variation.


### **Answer** :

<details><summary>CLICK ME</summary><p>0010</p></details>


### **Explaination** :

<details><summary>CLICK ME</summary><p>There are four components of a time series analysis: 
1. Secular trends—or simple trends.
2. Seasonal variations—or seasonal trends.
3. Cyclical fluctuations—or cyclical trends.
4. Irregular variations—or irregular trends.

A secular trend refers to the tendency of the series to increase, decrease, or stagnate over a long time. For example, a country's population could show an upward direction, while the number of death rates may show a downward trend. This trend is not seasonal or recurring.

On the other hand, a seasonal trend is a short-term fluctuation in a time series that occurs periodically. For example, sales during holidays trend much higher than during any other month, and the same happens every year.

A cyclical fluctuation is another variation that usually lasts for more than a year, and it's the effect of business cycles. Organizations go through these fluctuations in four different phases: prosperity, recession, depression, and recovery.

Finally, irregular variations are unpredictable fluctuations. We classify any variation that's not secular, seasonal, or cyclical as irregular. For example, we can't anticipate the effects of a hurricane on the economy.

The company here is dealing with a recession. Recessions are one of the phases of a cyclical fluctuation, so this was Payton's explanation. An important note is that cyclical variations do not have a fixed period—like seasonal variations do—but we can still predict them because we usually understand the sequence of changes that lead to these trends. 

In summary, the third choice is the correct answer to this question.</p></details>


### **References**: 

<details><summary>CLICK ME</summary><p>* [Introduction to Time Series Analysis](https://www.jigsawacademy.com/introduction-time-series-analysis/)
* [Definition of Time Series Analysis](https://www.toppr.com/guides/fundamentals-of-business-mathematics-and-statistics/time-series-analysis/definition-of-time-series-analysis/)</p></details>

-----------------------

