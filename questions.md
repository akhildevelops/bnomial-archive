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

## Date - 2022-05-12


## Title - A way to win Kaggle competitions


### **Question** :

Victoria joined Kaggle, and halfway through her first competition, she realized her single model wouldn't perform very well on the leaderboard. 

She learned that most people were using multiple models working together. Looking into that extra boost from ensembles was the only way she would be able to increase her score.

Victoria spent a couple of days reading about stacking and blending, two of the most popular techniques for building ensemble models. Although she was clear about the high-level idea, she wasn't sure about some of the differences between both techniques.

**Which of the following are valid differences between stacking and blending?**


### **Choices** :

- The meta-model created using stacking learns how to combine the predictions from multiple models. In contrast, the meta-model created using blending uses the predictions of the best contributing model.
- Stacking doesn't need models of comparable predictive power. In contrast, blending works like a weighted average and requires models to contribute positively to the ensemble.
- The meta-model created using stacking is trained on out-of-fold predictions made during cross-validation. In contrast, a meta-model created using blending is trained on predictions made on a holdout set.
- Stacking works well for both classification and regression problems. In contrast, Blending only works for regression problems.


### **Answer** :

<details><summary>CLICK ME</summary><p>0110</p></details>


### **Explaination** :

<details><summary>CLICK ME</summary><p>Victoria is right. Stacking and blending are powerful ensemble techniques and a must if you want to score high in Kaggle competitions.

Although both techniques have the same ultimate goal, there are essential differences in how they work. Let's start unraveling each of the available choices for this question to determine which ones are correct.

Stacking and blending use the concept of a "meta-model," a model that you train to average the results of other models. At a high level, both ensemble techniques use multiple models to generate predictions, and a meta-model to average those predictions and provide a final result.

The first choice argues that blending uses the predictions of the best contributing model, which is not true. Just like stacking, blending's meta-model uses the predictions of multiple models. Therefore, this is not a valid difference between both techniques.

An advantage of stacking is that it can benefit even from models that don't perform very well. In contrast, blending does require that models have a similar, good predictive power. Here is an excerpt from ["The Kaggle Book"](https://amzn.to/3kbanRb):

> (...) one interesting aspect of stacking is that you don't need models of comparable predictive power, as in averaging and often blending. In fact, even worse-performing models may be effective as part of a stacking ensemble. 

So even when using an individual model that performs poorly compared to all of the other models used by the stacking ensemble, the meta-model can use its out-of-fold predictions to improve its performance. Therefore, the second choice is a valid difference between both techniques.

Another valid difference between stacking and blending is the data they use to train the meta-model. The stacking meta-model is trained in the entire training set, using the [out-of-fold](https://machinelearningmastery.com/out-of-fold-predictions-in-machine-learning/) prediction strategy. The blending meta-model is trained in a holdout set that we randomly extract from the training dataset. Therefore, the third choice is also a valid difference between these techniques.

Finally, stacking and blending work well for regression and classification problems, so the final choice is incorrect.

In summary, the second and third choices are the correct answers to this question.</p></details>


### **References**: 

<details><summary>CLICK ME</summary><p>* [Stacking and Blending — An Intuitive Explanation](https://medium.com/@stevenyu530_73989/stacking-and-blending-intuitive-explanation-of-advanced-ensemble-methods-46b295da413c)
* [Stacking Ensemble Machine Learning With Python](https://machinelearningmastery.com/stacking-ensemble-machine-learning-with-python/)
* [Blending Ensemble Machine Learning With Python](https://machinelearningmastery.com/blending-ensemble-machine-learning-with-python/)
* [How to Use Out-of-Fold Predictions in Machine Learning](https://machinelearningmastery.com/out-of-fold-predictions-in-machine-learning/)
* [The Kaggle Book](https://amzn.to/3kbanRb)</p></details>

-----------------------

## Date - 2022-05-13


## Title - Pick the one you don't like


### **Question** :

Let's get straight to the point.

Your goal is to determine which of the following doesn't belong on this list.

**Can you select the odd one out?**


### **Choices** :

- AdaGrad
- RMSProp
- Adam
- SGD


### **Answer** :

<details><summary>CLICK ME</summary><p>0001</p></details>


### **Explaination** :

<details><summary>CLICK ME</summary><p>Every example here is an optimization method used when training a machine learning model. 

However, [AdaGrad](https://en.wikipedia.org/wiki/Stochastic_gradient_descent#AdaGrad), [RMSProp](https://en.wikipedia.org/wiki/Stochastic_gradient_descent#RMSProp), and [Adam](https://en.wikipedia.org/wiki/Stochastic_gradient_descent#Adam) are adaptive learning rate methods, while [SGD](https://en.wikipedia.org/wiki/Stochastic_gradient_descent) is not.

[Adaptive learning rate methods](https://towardsdatascience.com/learning-rate-schedules-and-adaptive-learning-rate-methods-for-deep-learning-2c8f433990d1) track and update different learning rates for each model parameter, while SGD uses the same learning rate for all parameters.

The last choice is the correct answer.</p></details>


### **References**: 

<details><summary>CLICK ME</summary><p>- [Learning Rate Schedules and Adaptive Learning Rate Methods for Deep Learning](https://towardsdatascience.com/learning-rate-schedules-and-adaptive-learning-rate-methods-for-deep-learning-2c8f433990d1)
- [Stochastic gradient descent: Extensions and variants](https://en.wikipedia.org/wiki/Stochastic_gradient_descent#Extensions_and_variants)
- [How to Configure the Learning Rate When Training Deep Learning Neural Networks](https://machinelearningmastery.com/learning-rate-for-deep-learning-neural-networks/)
- [Deep Learning](https://amzn.to/3CSjPkR)</p></details>

-----------------------

## Date - 2022-05-14


## Title - Depth perception


### **Question** :

Richard finally got a job as a self-driving car engineer!

His first task is to help the car perceive depth using the onboard cameras. He wants to start with an overview of the different approaches he can use to estimate the distance to every pixel in the image.

Before diving into the existing techniques, Richard has to think about the different ways he can capture pictures.

**Which of the following mechanisms do you think Richard can use to estimate depth?**


### **Choices** :

- An image from a single camera.
- A sequence of images from a single camera.
- A pair of images from a stereo camera.
- Cameras are 2D sensors, so Richard can't use them to estimate depth.


### **Answer** :

<details><summary>CLICK ME</summary><p>1110</p></details>


### **Explaination** :

<details><summary>CLICK ME</summary><p>Cameras are indeed 2D sensors, but there are many ways to estimate distance using pictures from a camera, so the last choice is incorrect.

Using a stereo camera is one of the classical approaches to do this. For every point observed in both camera images, we can triangulate its 3D position. Therefore, the third choice is correct.

We can also use a sequence of images from a single camera to triangulate fixed points over different frames. This method is called [Structure from Motion](https://en.wikipedia.org/wiki/Structure_from_motion) and is also a correct choice. I'd recommend listening to [Andrej Karpathy's talk](https://youtu.be/Ucp0TTmvqOE?t=8479) covering the work they are doing at Tesla to estimate depth using video.

The most interesting correct choice is the first one. We don't have enough information to triangulate the distance to a point in the image, but we can use our knowledge of the world to make some assumptions and solve the problem.

Remember, we are only interested in a car driving on the street, so we can exploit our understanding of the scene and our knowledge of standard dimensions of objects to estimate the distance to each point on the image. Over the last few years, we have seen several methods to train deep neural networks using this approach. ["Single Image Depth Estimation: An Overview"](https://arxiv.org/abs/2104.06456) is a good paper covering this topic.

In summary, every choice but the last one is correct.</p></details>


### **References**: 

<details><summary>CLICK ME</summary><p>- [Stereo Vision](https://en.wikipedia.org/wiki/Computer_stereo_vision)
- [Structure from Motion](https://en.wikipedia.org/wiki/Structure_from_motion)
- [Monocular depth estimation](https://paperswithcode.com/task/monocular-depth-estimation)
- [Depth from vision by Andrej Karpathy](https://youtu.be/Ucp0TTmvqOE?t=8479)
- [Single Image Depth Estimation: An Overview](https://arxiv.org/abs/2104.06456)
- [Multiple View Geometry](https://amzn.to/3KNPhmN)</p></details>

-----------------------

## Date - 2022-05-15


## Title - The benefits of the Huber loss


### **Question** :

The Huber loss is a popular loss function used for regression problems in machine learning.

[Here is the formula](https://en.wikipedia.org/wiki/Huber_loss). Take a second and look at it. 

The formula may look complex, but there are two things you need to know about the Huber loss. 

First, it behaves like a square function for values smaller than a parameter δ (similar to MSE.) Second, it acts as the absolute function for larger values (similar to MAE.)

In essence, the Huber loss is a combination of two other popular loss functions: Mean Squared Error (MSE) and Mean Absolute Error (MAE.)

**What are the benefits of combining these two functions?**


### **Choices** :

- It adds an additional hyperparameter δ which helps tune the model.
- It is more robust against large outliers than MSE.
- It is smooth around 0 helping the training converge better.
- It is continuous and differentiable.


### **Answer** :

<details><summary>CLICK ME</summary><p>0111</p></details>


### **Explaination** :

<details><summary>CLICK ME</summary><p>The [Huber loss](https://en.wikipedia.org/wiki/Huber_loss) tries to combine the advantages of both MSE and MAE. Here is a picture showing a comparison between these three functions (Image credit to ["Regression loss functions for machine learning"](https://www.evergreeninnovations.co/blog-machine-learning-loss-functions/)):

![image](https://user-images.githubusercontent.com/1126730/167011014-92c64b36-689e-4a89-bc6e-1e963807a982.png)

If we want to have a loss function that is not affected by outliers, we typically use MAE instead of MSE. When using MAE, we don't square the errors as we do with MSE, so outliers aren't amplified. However, MAE has the problem that it is not smooth around 0 (the derivative jumps a lot at 0,) which may cause issues with convergence. 

The Huber loss behaves like MAE for large values, so it's robust against outliers, but it acts like MSE around 0, so it is smooth. In a way, we get our cake and eat it too with the Huber loss! Therefore, the second and third options are correct.

An important goal for the Huber loss was to make it continuous and differentiable. This makes the fourth choice correct as well.

Finally, the Huber loss comes with an additional hyperparameter δ. That extra parameter means that the training process will be harder to tune. Although the parameter is essential in the design of the Huber loss, it's not an advantage compared to a loss function that doesn't require tuning. Therefore, the first choice is incorrect.

In summary, the second, third, and fourth choices are correct.</p></details>


### **References**: 

<details><summary>CLICK ME</summary><p>- [Huber loss](https://en.wikipedia.org/wiki/Huber_loss)
- [Huber Loss: Why Is It, Like How It Is?](https://www.cantorsparadise.com/huber-loss-why-is-it-like-how-it-is-dcbe47936473)
- [Regression loss functions for machine learning](https://www.evergreeninnovations.co/blog-machine-learning-loss-functions/)</p></details>

-----------------------

## Date - 2022-05-16


## Title - Climbing a hill


### **Question** :

Gabriela wanted her friend to grow an appreciation for the outdoors, so they started meeting every Saturday and going for a hike together.

And what better way to spend their time than starting a discussion about hill climbing and how it relates to their day-to-day work.

It turns out that hill climbing is an optimization algorithm that attempts to find a better solution by making incremental changes until it doesn't see further improvements.

Her friend couldn't help but notice how similar to gradient descent the process was, but Gabriela knew there were a few critical differences between them. 

**Can you select every correct statement from the following comparison list?**


### **Choices** :

- Hill climbing is a general optimization algorithm, but gradient descent is only used to optimize neural networks.
- Unlike gradient descent, hill climbing can return an optimal solution even if it's interrupted at any time before it ends.
- Gradient descent is usually more efficient than hill climbing, but there are fewer problems we can tackle with gradient descent.
- Both hill climbing and gradient descent can find optimal solutions for convex problems.


### **Answer** :

<details><summary>CLICK ME</summary><p>0011</p></details>


### **Explaination** :

<details><summary>CLICK ME</summary><p>[Hill climbing](https://en.wikipedia.org/wiki/Hill_climbing) is an optimization algorithm, just like [gradient descent](https://en.wikipedia.org/wiki/Gradient_descent) is. You can use both to minimize a function, regardless of whether it's related to a neural network, so the first choice is incorrect. For example, the following animation comes from ["Linear Regression using Gradient Descent."](https://towardsdatascience.com/linear-regression-using-gradient-descent-97a6c8700931) It shows a linear regression model that uses gradient descent as the optimization algorithm:

![Linear regression using gradient descent](https://miro.medium.com/max/1400/1*CjTBNFUEI_IokEOXJ00zKw.gif)

The second choice is also incorrect. Hill climbing can return a valid solution even if it's interrupted before it ends, but there's no guarantee that this will be the optimal solution. We call these types of algorithms ["anytime algorithms."](https://en.wikipedia.org/wiki/Anytime_algorithm) They find better and better solutions the longer they keep running but can return a reasonable solution at any time.

Gradient descent looks at the slope in the local neighborhood and moves in the direction of the steepest slope. This makes it much more efficient than hill climbing, which needs to look at all neighboring states to evaluate the cost function in each of them. 

The efficiency gained by gradient descent presents a trade-off: the algorithm assumes that you can compute the function's gradient in any given state, limiting the problems where we can use it. Therefore, the third choice is correct.

Finally, the fourth choice is also a correct answer. Both algorithms can find the optimal solution for a convex problem. Look at the following example of a [convex function](https://en.wikipedia.org/wiki/Convex_function). Assuming we configure hill climbing to optimize for finding the minimum, neither function should have trouble getting all the way to the bottom of this problem:

![A convex function](https://user-images.githubusercontent.com/1126730/167182794-30f47b44-2149-4700-b642-616d8d6dce51.png)

In summary, the third and fourth choices are the correct answers to this question.</p></details>


### **References**: 

<details><summary>CLICK ME</summary><p>* [Hill Climbing Algorithms (and gradient descent variants) IRL](https://umu.to/blog/2018/06/29/hill-climbing-irl)
* [Linear Regression using Gradient Descent](https://towardsdatascience.com/linear-regression-using-gradient-descent-97a6c8700931)
* [Hill climbing](https://en.wikipedia.org/wiki/Hill_climbing)
* [Gradient descent](https://en.wikipedia.org/wiki/Gradient_descent)</p></details>

-----------------------

## Date - 2022-05-17


## Title - Which function is she using?


### **Question** :

Kiara was leaving her team, but she didn't want to go without having some fun.

She put together a simple neural network with one hidden layer. Never trained it, but she initialized its parameters and told her team that they should expect every node from the hidden layer to return a value resembling the following formula:

```
y = max(0.01 * x, 0)
```

Kiara saved the model and asked her team to test the node results without looking at the code. They found out that, in effect, the results always followed the formula mentioned by Kiara.

Kiara's question to her team was simple:

**Which of the following activation functions am I using in this network?**


### **Choices** :

- Kiara is using the sigmoid activation function.
- Kiara is using the Rectified Linear Unit activation function.
- Kiara is using the Leaky Rectified Linear Unit activation function.
- None of the above activation functions can produce this output.


### **Answer** :

<details><summary>CLICK ME</summary><p>0100</p></details>


### **Explaination** :

<details><summary>CLICK ME</summary><p>This is a fun, interesting question and one where we need to be very careful to find the correct answer.

The team doesn't have access to the network architecture, so all they know is that node outputs follow a specific pattern. They also know Kiara is using an activation function. If we use σ to represent this activation function, the result of each node should look like this:

```
z = σ(y)
```

Here, `z` is the output the team is getting out of the node, and `y` is the input to the activation function. This input results from `y = w * x + b`, where `b` is the bias, and `w` is the weight assigned to that node. Putting everything together:

```
z = σ(w * x + b)
```

Let's start with [sigmoid](https://en.wikipedia.org/wiki/Sigmoid_function), which doesn't use the `max` operation, so we can safely discard it.

Here is the formula of the [Leaky Rectified Linear Unit](https://paperswithcode.com/method/leaky-relu) (Leaky ReLU):

```
y = max(0.01 * x, x)
```

This one looks promising, and it's how Kiara wanted to prank her team. Assuming that `x` results from `w * x + b`, Leaky ReLU would almost make sense, except it returns the maximum between a scaled version of `x` and `x`, while the team is seeing something different. Leaky ReLU can't possibly be the answer.

Here is the formula of the [Rectified Linear Unit](https://en.wikipedia.org/wiki/Rectifier_(neural_networks)) (ReLU):

```
y = max(x, 0)
```

It looks similar, but where is the scaling factor? Well, Kiara initialized the network, so there's a good chance she did it in a way to confuse everyone. If Kiara set every weight `w` to be `0.01` and every bias term to be zero, we would get the following:

```
z = σ(w * x + b)
z = σ(0.01 * x + 0)
```

Assuming that σ is the ReLU activation function, we will get the following:

```
z = max(0.01 * x + 0, 0)
z = max(0.01 * x, 0)
```

This is the pattern the team is seeing. Kiara used ReLU as her activation function.</p></details>


### **References**: 

<details><summary>CLICK ME</summary><p>* [A Gentle Introduction to the Rectified Linear Unit (ReLU)](https://machinelearningmastery.com/rectified-linear-activation-function-for-deep-learning-neural-networks/)
* [Rectifier (neural networks)](https://en.wikipedia.org/wiki/Rectifier_(neural_networks))
* [Leaky ReLU](https://paperswithcode.com/method/leaky-relu)
* [Activation Functions](https://himanshuxd.medium.com/activation-functions-sigmoid-relu-leaky-relu-and-softmax-basics-for-neural-networks-and-deep-8d9c70eed91e)</p></details>

-----------------------

## Date - 2022-05-18


## Title - Riley's speed-dating match


### **Question** :

If you spend all day sitting at a desk, you can't expect to have many opportunities to meet interesting people.

Riley decided to get to bull by the horns and checked in on one of those speed-dating sites that promise to find your perfect match.

But of course, Silicon Valley is a ridiculous caricature of the impossible, and Riley's first match decided to start blabbing about machine learning and dimensionality reduction algorithms.

And if this wasn't crazy enough, Riley didn't think this person knew what he was talking about.

**Can you guess all the possible statements about dimensionality reduction that would make Riley's match incorrect?**


### **Choices** :

- Supervised learning algorithms can be used as dimensionality reduction techniques.
- Every dimensionality reduction technique is a clustering technique, but every clustering technique is not a dimensionality reduction algorithm.
- Dimensionality reduction algorithms are primarily considered unsupervised learning techniques.
- Nowadays, the most successful dimensionality reduction techniques are deep learning algorithms.


### **Answer** :

<details><summary>CLICK ME</summary><p>0101</p></details>


### **Explaination** :

<details><summary>CLICK ME</summary><p>Here is something clear to Riley: Dimensionality reduction algorithms reduce the number of input variables in a dataset to find a lower-dimensional representation that still preserves the salient relationships in the data.

For example, PCA—short for [Principal Component Analysis](https://en.wikipedia.org/wiki/Principal_component_analysis)—is a dimensionality reduction algorithm often used to reduce the number of variables in a dataset while preserving as much information as possible. Another dimensionality reduction technique is [Independent Component Analysis](https://en.wikipedia.org/wiki/Independent_component_analysis) (ICA).

Everywhere you go, dimensionality reduction algorithms are classified as unsupervised learning techniques. Even auto-encoders that require training a neural network are not considered a supervised technique, as mentioned in ["Machine Learning: A Probabilistic Perspective"](https://amzn.to/3s39PRD):

> An auto-encoder is a kind of unsupervised neural network that is used for dimensionality reduction and feature discovery. More precisely, an auto-encoder is a feedforward neural network that is trained to predict the input itself.

This doesn't mean that you can't use a supervised learning method to reduce the dimensionality of a dataset. For example, here is an excerpt from ["Seven Techniques for Data Dimensionality Reduction"](https://www.knime.com/blog/seven-techniques-for-data-dimensionality-reduction):

> Decision Tree Ensembles, also referred to as random forests, are useful for feature selection in addition to being effective classifiers. One approach to dimensionality reduction is to generate a large and carefully constructed set of trees against a target attribute and then use each attribute's usage statistics to find the most informative subset of features.

At this point, we know that the first and the third choices are correct statements about dimensionality reduction. But what about the other two options?

The second choice is incorrect because every dimensionality reduction technique is not a clustering technique. For example, neither PCA nor ICA are clustering methods.

The fourth option is also incorrect because it's not true that the most successful dimensionality reduction techniques are limited to deep learning algorithms. For example, PCA is one of the most popular dimensionality reduction techniques and has nothing to do with deep learning.

If Riley's match was incorrect, he must have mentioned the second or fourth statements, so they are the correct answer to this question.</p></details>


### **References**: 

<details><summary>CLICK ME</summary><p>* [Introduction to Dimensionality Reduction for Machine Learning](https://machinelearningmastery.com/dimensionality-reduction-for-machine-learning/)
* [Machine Learning: A Probabilistic Perspective](https://amzn.to/3s39PRD)
* [Seven Techniques for Data Dimensionality Reduction](https://www.knime.com/blog/seven-techniques-for-data-dimensionality-reduction)
* [A Gentle Introduction to LSTM Autoencoders](https://machinelearningmastery.com/lstm-autoencoders/)</p></details>

-----------------------

## Date - 2022-05-19


## Title - Occam's Razor showoff


### **Question** :

Tiara's manager was a showoff. No matter the situation, he always found a way to show everyone how smart he was.

Tiara noticed that he's been getting into machine learning lately, and as cringe as it sounds, he has been using "Occam's Razor" on every occasion, even incorrectly.

Tiara started a secret list collecting every scenario when her manager used Occam's Razor to explain a situation. At the end of the week, she sent it to many of her friends to have a good laugh.

**Which of the following situations from Tiara's list are you comfortable justifying with Occam's Razor?**


### **Choices** :

- We should prefer simpler models with fewer coefficients over complex models like ensembles.
- Feature selection and dimensionality reduction help simplify models to get better results.
- Keeping the training process as fast as possible avoids overtraining and prevents overcomplicated results.
- Starting the training of the model using sensible values for the hyperparameters.


### **Answer** :

<details><summary>CLICK ME</summary><p>1100</p></details>


### **Explaination** :

<details><summary>CLICK ME</summary><p>[Occam's Razor](https://en.wikipedia.org/wiki/Occam%27s_razor) is a principle that says that if you have two competing ideas to explain the same phenomenon, you should prefer the simpler one.

There are a couple of situations in this list where using Occam's Razor is a stretch. The third choice is probably the simplest one to tackle first: it talks about "the speed of the training process" and relates it to overtraining and overcomplicating results. Not only does this has nothing to do with Occam's Razor, but a quick training process doesn't necessarily reduce complexity. 

The fourth choice is also not correct. Starting training using sensible values for the hyperparameters is essential, but we can't explain this using Occam's Razor.

Occam's Razor fits the first choice like a glove. Given two learning algorithms with similar tradeoffs, we should use the least complex and most straightforward to interpret. At least this time, Tiara's boss was correct.

Finally, the second choice is not an obvious fit, but we could argue it's also a correct answer. Feature selection and dimensionality reduction simplify the data we use to train our models. We use these steps to remove redundant or irrelevant information, therefore getting a simpler dataset that should perform better than a more complex one.

In summary, Tiara's manager was correct on the first two but was incorrect on the last two.</p></details>


### **References**: 

<details><summary>CLICK ME</summary><p>* [Ensemble Learning Algorithm Complexity and Occam’s Razor](https://machinelearningmastery.com/ensemble-learning-and-occams-razor/)
* [How does Occam's razor apply to machine learning?](https://www.techopedia.com/how-does-occams-razor-apply-to-machine-learning/7/33087)
* The [Occam's razor](https://en.wikipedia.org/wiki/Occam%27s_razor) definition in Wikipedia.</p></details>

-----------------------

## Date - 2022-05-20


## Title - Emma's list


### **Question** :

Nothing is perfect.

And no matter how much they said otherwise, Emma knew that gradient descent was no exception.

They have been discussing some of the most popular optimization algorithms for neural networks, and the team didn't want to listen despite Emma's comments regarding some of the downsides of gradient descent.

Emma decided to post a detailed list of problems on the company's Slack channel.

**Which of the following practical issues of gradient descent deserve to be on Emma's list?**


### **Choices** :

- Gradient descent can take a long time to converge to a local minimum.
- There's no guarantee that gradient descent will converge to the global minima.
- Gradient descent is susceptible to the initialization of the network's weights.
- Gradient descent is not capable of optimizing continuous functions.


### **Answer** :

<details><summary>CLICK ME</summary><p>1110</p></details>


### **Explaination** :

<details><summary>CLICK ME</summary><p>Gradient descent is one of the most popular optimization algorithms used in machine learning applications. But, despite its popularity, there are several practical issues that Emma wanted to mention.

The first issue is how gradient descent updates the model parameters after calculating the derivatives for all the observations. When working with large datasets, finding a local minimum may take a long time because the algorithm needs to compute many gradients before making a single update. [Stochastic Gradient Descent](https://en.wikipedia.org/wiki/Stochastic_gradient_descent) (SGD), a variation of gradient descent, works differently and updates the model parameters for each observation speeding up the process. Since the team is focusing on gradient descent, the first choice made it to Emma's list.

The second choice is also on the list. Assuming the are multiple local minima in a problem, there is no guarantee that gradient descent will find the global minimum. Here is an excerpt from ["Gradient Descent,"](http://www.cs.umd.edu/~djacobs/CMSC426/GradientDescent.pdf) a publication from the Computer Science Department of the University of Maryland:

> When a problem is nonconvex, it can have many local minima. And depending on where we initialize gradient descent, we might wind up in any of those local minima since they are all fixed points.

But it doesn't end there. As the previous quote mentions, gradient descent is also susceptible to the initialization of the network's weights. Assuming there are multiple local minima, the initialization of the network weights will play a fundamental role in whether the algorithm finds the global minimum: it may converge to a less optimal solution if we initialize the network too far from the global minimum. Therefore, the third choice is also a correct answer.

Finally, gradient descent can optimize a continuous function with no issues, so the fourth choice is not a correct answer.

In summary, Emma included the first three choices on her list.</p></details>


### **References**: 

<details><summary>CLICK ME</summary><p>* [Gradient Descent](http://www.cs.umd.edu/~djacobs/CMSC426/GradientDescent.pdf) is a deep dive into gradient descent and its variants from the Computer Science Department of the University of Maryland.
* [Problems with Gradient Descent](https://www.encora.com/insights/problems-with-gradient-descent)
* [Gradient Descent For Machine Learning](https://machinelearningmastery.com/gradient-descent-for-machine-learning/)</p></details>

-----------------------

## Date - 2022-05-21


## Title - Zoe's looking into KNN


### **Question** :

It was the first time Zoe dealt with k-Nearest Neighbors (KNN). She inherited the code, and now she was responsible for making it work.

Before touching the code, she decided to do some research. Her first stop was on one of the fundamental topics in machine learning: bias, variance, and their relationship with the algorithm.

She knows there's always a tradeoff between these two.

**Which of the following statements are correct concerning the bias and variance tradeoff of KNN?**


### **Choices** :

- Zoe can increase the bias of KNN by using a larger value of `k`.
- Zoe can increase the variance of KNN by using a larger value of `k`.
- Zoe can decrease the bias of KNN by using a smaller value of `k`.
- Zoe can decrease the variance of KNN by using a smaller value of `k`.


### **Answer** :

<details><summary>CLICK ME</summary><p>1010</p></details>


### **Explaination** :

<details><summary>CLICK ME</summary><p>[KNN](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm) is an algorithm with low bias and high variance. 

Let's imagine Zoe decides to use a small value of `k`, for example, `k=1`. In this case, the algorithm will likely predict the training dataset perfectly. The smaller the value of `k`, the less bias and larger variance KNN will show. This, of course, is not a great outcome because the model will overfit and have difficulty predicting unseen data.

Now let's assume Zoe decides to use a very large value of `k`; for example, set `k` to the number of samples on her training dataset. This will increase the algorithm's bias and reduce its variance, resulting in an underfit model that can't adequately capture the variance in the training dataset. Here is a quote from ["Why Does Increasing k Decrease Variance in kNN?"](https://towardsdatascience.com/why-does-increasing-k-decrease-variance-in-knn-9ed6de2f5061):

> If we take the limit as k approaches the size of the dataset, we will get a model that just predicts the class that appears more frequently in the dataset [...]. This is the model with the highest bias, but the variance is 0 [...]. High bias because it has failed to capture any local information about the model, but 0 variance because it predicts the exact same thing for any new data point.

As Zoe suspects, neither case will lead to a proper solution. She needs to find the appropriate tradeoff between the bias and variance of the algorithm.

In summary, the smaller the value of `k` is, the lower the bias and higher the variance. The larger the value of `k` is, the higher the bias and lower the variance. This means that the first and third choices are correct: we can control the algorithm's bias as explained in these two choices.</p></details>


### **References**: 

<details><summary>CLICK ME</summary><p>* ["Why Does Increasing k Decrease Variance in kNN?"](https://towardsdatascience.com/why-does-increasing-k-decrease-variance-in-knn-9ed6de2f5061) is a really good article diving into the relationship of `k` and the variance of KNN.
* For a more general introduction to the bias-variance trade-off, check ["Gentle Introduction to the Bias-Variance Trade-Off in Machine Learning"](https://machinelearningmastery.com/gentle-introduction-to-the-bias-variance-trade-off-in-machine-learning/).
* In case you prefer Twitter threads with a summary of how this works, check out ["Bias, variance, and their relationship with machine learning algorithms."](https://twitter.com/svpino/status/1506964069646884864)</p></details>

-----------------------

## Date - 2022-05-22


## Title - The 3-sigma accuracy


### **Question** :

Clara and her team are working on a drone localization project.

They have developed a neural network model that uses drone cameras to determine its position in the world so the drone can come back and land at the same spot it took off.

Clara was discussing the latest evaluation results with her colleagues when Jan mentioned that their latest model reached a 3-sigma accuracy of 20cm. 

Clara is new to the industry, and _"3-sigma accuracy of 20cm"_ didn't make much sense.

**What does a "3-sigma accuracy of 20cm" mean in this context?**


### **Choices** :

- In 66.6% of the cases, the model's error is less than 20cm
- In 68.2% of the cases, the model's error is less than 20cm
- In 95.4% of the cases, the model's error is less than 20cm
- In 99.7% of the cases, the model's error is less than 20cm


### **Answer** :

<details><summary>CLICK ME</summary><p>0001</p></details>


### **Explaination** :

<details><summary>CLICK ME</summary><p>The 3-sigma accuracy is a common way to quantify the accuracy of a model when estimating a continuous variable. Here is a quote from [Wikipedia's explanation of the 68 - 95 - 99.7 rule](https://en.wikipedia.org/wiki/68%E2%80%9395%E2%80%9399.7_rule):

> In the empirical sciences, the so-called three-sigma rule of thumb (or 3σ rule) expresses a conventional heuristic that nearly all values are taken to lie within three standard deviations of the mean, and thus it is empirically useful to treat 99.7% probability as near certainty.

We can often assume that the error of estimating a continuous variable (such as the drone's position) follows the normal distribution. If we denote the standard deviation of the normal distribution as σ (sigma), then 68.2% of the samples should fall in the region from -1σ to 1σ around the mean.

![Standard deviation](https://user-images.githubusercontent.com/1126730/169593614-ee0ecdf7-a262-41b3-943f-5ae6865afcc8.png)

If we take a larger range from -2σ to 2σ, then 95.4% of a normally distributed dataset will fall in this interval. Finally, a 3σ interval will cover 99.73% of the samples.

Therefore, when we talk about a 3-sigma accuracy of 20cm, we mean that 99.73% of the model predictions are more accurate than 20cm (because they fall in the -3σ to 3σ interval). Thus, the correct answer is the fourth choice.</p></details>


### **References**: 

<details><summary>CLICK ME</summary><p>- [Normal distribution](https://en.wikipedia.org/wiki/Normal_distribution#Standard_deviation_and_coverage)
- [68–95–99.7 rule](https://en.wikipedia.org/wiki/68%E2%80%9395%E2%80%9399.7_rule)</p></details>

-----------------------

## Date - 2022-05-23


## Title - Scheduled learning


### **Question** :

A company—a bad one, because there are plenty of those out there—has been experiencing some turnaround, and they wanted to ensure the new team members were up to speed with the neural network model they were using in production.

The team has been looking at the code and writing notes every time they find something new.

They stumbled upon the training scripts and noticed the last team used a learning rate scheduler.

**Which of the following statements could explain why the last team used this scheduler? Select all that apply.**


### **Choices** :

- The last team used the learning rate scheduler to increase the learning rate as training progressed.
- The last team used the learning rate scheduler to decrease the learning rate as training progressed.
- The last team used the learning rate scheduler to give the network a better chance to converge.
- The last team used the learning rate scheduler to save the learning rate at specific intervals during the training process.


### **Answer** :

<details><summary>CLICK ME</summary><p>0110</p></details>


### **Explaination** :

<details><summary>CLICK ME</summary><p>When training a neural network, setting the hyperparameters of the optimizer is essential for getting good results. One of the most critical parameters is the [learning rate](https://en.wikipedia.org/wiki/Learning_rate). Setting the learning rate too high or too low will cause problems during training.

A simple way to think about the learning rate is as follows: if we set it too low, the training process will be very slow; it will take a long time for the network to converge. Conversely, if we use a learning rate that's too high, the process will oscillate around the minimum without converging. Here is a chart from ["Deep Learning Wizard"](https://www.deeplearningwizard.com/deep_learning/boosting_models_pytorch/lr_scheduling/) illustrating the effect of different learning rates:

![Differences in learning rates](https://user-images.githubusercontent.com/1126730/167927199-f6a2add7-91be-4bc6-8459-bf00ff0ea4b6.png)

A popular technique to find a good balance is to use a learning rate scheduler. This predefined schedule adjusts the learning rate between epochs or iterations as the training progresses.

The most common scenario is to start with a high learning rate and decrease it over time. In the beginning, we take significant steps towards the minimum but move more carefully as we hone in on it. 

Looking at the available choices, we can see the first choice is incorrect, but the second and third choices are correct. The team was likely trying to decrease the learning rate as training progressed. Although there are experiments showing the use of [cyclical learning rates](https://arxiv.org/abs/1506.01186), the most common practice when using a scheduler is to start with a high learning rate and reduce it over time.

Finally, the fourth choice is incorrect as well. A learning rate scheduler has nothing to do with saving the learning rate. 

In summary, the second and third choices are the correct answers to this question.</p></details>


### **References**: 

<details><summary>CLICK ME</summary><p>- ["Learning Rate Scheduling"](https://d2l.ai/chapter_optimization/lr-scheduler.html) is a great introduction to learning rate schedulers.
- ["How to Choose a Learning Rate Scheduler for Neural Networks?"](https://neptune.ai/blog/how-to-choose-a-learning-rate-scheduler) is an article from [Neptune AI](https://neptune.ai/), focusing on some practical ideas on how to use schedulers.
- ["Cyclical Learning Rates for Training Neural Networks"](https://arxiv.org/abs/1506.01186) is the paper discussing a technique to let the learning rate cyclically vary between reasonable boundary values.</p></details>

-----------------------

## Date - 2022-05-24


## Title - Balancing bias and variance


### **Question** :

The very first chapter of her machine learning book was about bias, variance, and their tradeoff. 

Callie knew that she had no alternative: she had to spend the time trying to understand these concepts before moving on.

But at the end of the day, it wasn't easy to remember the nuances of each concept, so Callie decided to get help.

**Which of the following descriptions of bias and variance are correct?**


### **Choices** :

- Bias refers to the assumptions a model makes to simplify the process of finding answers. The more assumptions it makes, the more biased the model is.
- Variance refers to the assumptions a model makes to simplify finding answers. The more assumptions it makes, the more variance in the model.
- Bias refers to how much the answers given by the model will change if we use different training data. The model has low bias if the answers stay the same regardless of the data.
- Variance refers to how much the answers given by the model will change if we use different training data. The model has low variance if the answers stay the same regardless of the data.


### **Answer** :

<details><summary>CLICK ME</summary><p>1001</p></details>


### **Explaination** :

<details><summary>CLICK ME</summary><p>Every machine learning algorithm deals with three types of errors: 
* Bias error 
* Variance error 
* Irreducible error 

To answer this question, let's forget about the irreducible error and focus on the other two.

Here is what [Jason Brownlee](https://machinelearningmastery.com/gentle-introduction-to-the-bias-variance-trade-off-in-machine-learning/) has to say about _bias_:
> Bias are the simplifying assumptions made by a model to make the target function easier to learn.

Think about a simple linear model. It assumes that the target function is linear, so the model will try to fit a line through the data regardless of its appearance. This assumption helps the model simplify the process of finding the answer, and the more assumption it makes, the more biased the model is. Often, linear models are high-bias, and nonlinear models are low-bias.

Here is a [funny depiction of biases](https://xkcd.com/2618/): the speaker believes everyone must understand selection bias. Whenever we put people in buckets to characterize or predict how they act, we use our biases to simplify our understanding of the world.

![An example of selection bias](https://user-images.githubusercontent.com/1126730/168139417-8e5d8ce5-929e-4f8f-96a1-80232d61c73e.png)

Regarding variance, [Jason](https://machinelearningmastery.com/gentle-introduction-to-the-bias-variance-trade-off-in-machine-learning/) continues:
> Variance is the amount that the estimate of the target function will change if different training data is used.

Variance refers to how much the answers given by the model will change if we use different training data. The model has low variance if the answers stay the same regardless of the data. 

Think about a fickle person that constantly changes their mind with the news. Every new article makes the person believe something completely different. I don't want to overextend the analogy, but this is an example of high variance. Often, linear models are low-variance, and nonlinear models are high-variance.

If we consider all of this, the first and fourth choices are the correct answer to this question.</p></details>


### **References**: 

<details><summary>CLICK ME</summary><p>* Here is Jason Brownlee's article I mentioned before: ["Gentle Introduction to the Bias-Variance Trade-Off in Machine Learning"](https://machinelearningmastery.com/gentle-introduction-to-the-bias-variance-trade-off-in-machine-learning/).
* The Wikipedia page on bias and variance is also a good resource: ["Bias–variance tradeoff"](https://en.wikipedia.org/wiki/Bias–variance_tradeoff).
* In case you like the simplicity of Twitter threads, here is one for you about this topic: ["Bias, variance, and their relationship with machine learning algorithms"](https://twitter.com/svpino/status/1390969728504565761).</p></details>

-----------------------

## Date - 2022-05-25


## Title - Cutting down features


### **Question** :

When Nicole finished collecting the data, she realized that there were just too many features.

She was staring at hundreds of potential variables, and it was evident that any model would have a hard time navigating them. Nicole knew that she had to reduce the dimensionality of her dataset.

Dimensionality reduction algorithms reduce the number of input variables in a dataset to find a lower-dimensional representation that still preserves the salient relationships in the data.

**Which of the following are dimensionality reduction techniques that Nicole could use?**


### **Choices** :

- Singular Value Decomposition
- Principal Component Analysis
- Linear Discriminant Analysis
- Isomap Embedding


### **Answer** :

<details><summary>CLICK ME</summary><p>1111</p></details>


### **Explaination** :

<details><summary>CLICK ME</summary><p>Every one of these is a valid dimensionality reduction technique.

The problem doesn't specify the type of data that Nicole is using, so it's hard to determine which of these techniques will be most effective, but every one of them could be potentially valuable.

Therefore, all choices are correct.</p></details>


### **References**: 

<details><summary>CLICK ME</summary><p>* [Singular value decomposition](https://en.wikipedia.org/wiki/Singular_value_decomposition)
* [Understanding Dimension Reduction with Principal Component Analysis (PCA)](https://blog.paperspace.com/dimension-reduction-with-principal-component-analysis/)
* [Linear Discriminant Analysis – Bit by Bit](https://sebastianraschka.com/Articles/2014_python_lda.html)
* [Dimension Reduction - IsoMap](https://blog.paperspace.com/dimension-reduction-with-isomap/)</p></details>

-----------------------

## Date - 2022-05-26


## Title - But why deep learning?


### **Question** :

Martin met an old friend for a coffee.

They discussed Martin's latest work on image classification using deep learning. While Martin's friend used to work in computer vision 12 years ago, he is not aware of any of the latest developments.

He asks Martin why deep learning is so successful in image classification compared to the classical computer vision methods. He's heard that deep learning is better but doesn't understand why.

**If Martin wanted to summarize his reasoning with one sentence, which of the following is the best way to explain why deep learning is better for computer vision tasks?**


### **Choices** :

- Deep neural networks have many fully connected layers making the model more powerful.
- Deep neural networks can learn the best features for the task, while traditional methods rely on engineered features.
- Deep learning methods can use much larger datasets and achieve better performance.
- Deep learning algorithms can take advantage of GPUs, and we can therefore train much larger and more powerful models.


### **Answer** :

<details><summary>CLICK ME</summary><p>0100</p></details>


### **Explaination** :

<details><summary>CLICK ME</summary><p>These options explain why deep learning methods can achieve good results on computer vision tasks, but not all explain why deep learning is better than classical computer vision methods.

Having more data and algorithms optimized for GPUs can improve the results of a model. However, this is not exclusive to deep learning methods. Many traditional machine learning models can handle large datasets and take advantage of GPUs. Therefore, neither the third nor fourth choices correctly explain why deep learning is better for computer vision tasks.

The first choice is also incorrect. Although deep learning models can use many fully connected layers, the main benefits when solving computer vision problems come from using specialized layers—like convolutional layers—and not fully connected ones.

Finally, traditional computer vision methods typically rely on pre-computed features. Contrast this with [Convolutional Neural Networks](https://en.wikipedia.org/wiki/Convolutional_neural_network) and [Vision Transformers](https://en.wikipedia.org/wiki/Vision_transformer), which can learn features directly from the dataset and don't need pre-computed features to provide good results.

The ability to learn powerful features is one of the main reasons for the superior performance of deep learning methods in computer vision, making the second choice the best explanation for Martin's friend.</p></details>


### **References**: 

<details><summary>CLICK ME</summary><p>- Here is an introduction to ["Convolutional Neural Networks"](https://en.wikipedia.org/wiki/Convolutional_neural_network).
- And this is the introduction to ["Vision Transformers"](https://en.wikipedia.org/wiki/Vision_transformer).
- ["Learned Features"](https://christophm.github.io/interpretable-ml-book/cnn-features.html) is an excellent explanation covering the features that we can learn using convolutional layers.
- Check out ["OpenAI Microscope"](https://openai.com/blog/microscope/) for a fascinating look at the visual features inside a neural network.</p></details>

-----------------------

## Date - 2022-05-27


## Title - Choosing a loss function


### **Question** :

Ariana and Zach need to compute how different their model predictions are from the expected results.

They have been going back and forth between two different loss functions: Root Mean Squared Error (RMSE) and Mean Absolute Error (MAE). These two metrics have properties that will shine depending on the problem they want to solve.

**Which of the following is the correct way to think about these two metrics?**


### **Choices** :

- RMSE penalizes larger differences between the predictions and the expected results.
- RMSE is significantly faster to compute than MAE.
- From both metrics, RMSE is the only one indifferent to the direction of the error.
- From both metrics, MAE is the only one indifferent to the direction of the error.


### **Answer** :

<details><summary>CLICK ME</summary><p>1000</p></details>


### **Explaination** :

<details><summary>CLICK ME</summary><p>When we train a machine learning model, we need to compute how different our predictions are from the expected results. For example, if we predict a house's price as `$150,000`, but the correct answer is `$200,000`, our "error" is `$50,000`.

There are multiple ways we can compute this error, but two common choices are:
* RMSE — [Root Mean Squared Error](https://en.wikipedia.org/wiki/Root-mean-square_deviation)
* MAE — [Mean Absolute Error](https://en.wikipedia.org/wiki/Mean_absolute_error)

These have different properties that will shine depending on the problem we want to solve. Remember that the optimizer will use this error to adjust the model. We want to set up the right incentives so the model learns appropriately.

Let's focus on a critical difference between these two metrics. Remember the "squared" portion of the RMSE? You are "squaring" the difference between the prediction and the expected value. Why is this relevant?

Squaring the difference "penalizes" larger values. If you expect a prediction to be 2, but you get 10, using RMSE, the error will be (2 - 10)² = 64. However, if you get 5, the error will be (2 - 5)² = 9. Do you see how it penalizes larger errors?

MAE doesn't have the same property. The error increases proportionally with the difference between predictions and target values. Understanding this is important to decide which metric is better for each case. 

Predicting a house's price is a good example where `$10,000` off is twice as bad as `$5,000`. We don't necessarily need to rely on RMSE here, and MAE may be all we need. 

But predicting the pressure of a tank may work differently. While 5 psi off may be within the expected range, 10 psi off may be a complete disaster. Here "10" is much worse than just two times "5", so RMSE may be a better solution.

Looking at the first choice, we already know it is a correct answer. RMSE penalizes larger differences between predictions and expected results.

Looking at both formulas, RMSE has extra squaring and root squaring operations, so it can't be faster to compute than MAE. The second choice is, therefore, not correct.

The third choice states that RSME is indifferent to the direction of the error, but MAE isn't. This is not correct: MAE uses the absolute value of the error, so both negative and positive values will end up being the same.

The fourth choice states that MAE is indifferent to the direction of the error, but RMSE isn't. This is not correct either: RMSE squares the error, so both negative and positive values will be the same.

In summary, the only correct answer to this question is the first choice.</p></details>


### **References**: 

<details><summary>CLICK ME</summary><p>* ["RMSE vs MAE, which should I use?"](https://stephenallwright.com/rmse-vs-mae/) this is a great summary by Stephen Allwright about the properties of these two functions and how you should think about them.
* ["Root-mean-square deviation"](https://en.wikipedia.org/wiki/Root-mean-square_deviation) is the Wikipedia page covering RMSE.
* ["Mean absolute error"](https://en.wikipedia.org/wiki/Mean_absolute_error) is the Wikipedia page covering MAE.</p></details>

-----------------------

## Date - 2022-05-28


## Title - Rolling down the hill


### **Question** :

Brooklyn was dealing with a complex problem. Although gradient descent was working relatively well, she read that adding momentum could benefit her use case.

Brooklyn needed to justify spending more time on this problem, so she wrote an email summarizing her reasoning behind using momentum, hit send, and patiently waited for her manager to respond.

**Which of the following statements are some of the reasons that Brooklyn included in her email?**


### **Choices** :

- Momentum helps when there's a lot of variance in the gradients.
- Momentum helps overcome local minima.
- Momentum helps the training process converge faster.
- Momentum helps when there aren't flat regions in the search space.


### **Answer** :

<details><summary>CLICK ME</summary><p>1110</p></details>


### **Explaination** :

<details><summary>CLICK ME</summary><p>There's a problem with [gradient descent](https://en.wikipedia.org/wiki/Gradient_descent): it depends entirely on the gradients it computes along the way, so whenever there's a lot of variance in these gradients, the algorithm can bounce around the search space making the optimization process slower.

Adding [momentum](https://en.wikipedia.org/wiki/Stochastic_gradient_descent#Momentum) to gradient descent will help overcome this problem. Here is Jason Brownlee on ["Gradient Descent With Momentum from Scratch"](https://machinelearningmastery.com/gradient-descent-with-momentum-from-scratch/):

> Momentum involves adding an additional hyperparameter that controls the amount of history (momentum) to include in the update equation, i.e. the step to a new point in the search space. 

This parameter will help gradient descent accelerate in one direction based on past updates. A good analogy is a ball rolling down the hill. The more momentum it gains, the faster the ball will move in the direction of travel. If we have noisy gradients, momentum will help dampen the noise and keep the algorithm moving in the correct direction. Therefore, the first choice is correct.

This explanation also helps understand why the third choice is correct as well. Having gradients with a lot of variance will cause gradient descent to spend a long time bouncing around, while adding momentum will straighten the direction of the search. This will lead to faster convergence.

Momentum helps the optimization overcome small local minima by rolling past them. Going back to our example of a ball rolling down the hill, the more momentum it has, the more likely it will be to overcome small dips in the ground. This makes the second choice correct as well.

Finally, the fourth choice is not correct because momentum does help with flat regions in the search space. In the same way it can overcome small dips in the surface, momentum can help gradient descent get past a flat region by continuing its previous movement. Here is [Jason Brownlee](https://machinelearningmastery.com/gradient-descent-with-momentum-from-scratch/) again:

> (...) momentum is helpful when the search space is flat or nearly flat, e.g. zero gradient. The momentum allows the search to progress in the same direction as before the flat spot and helpfully cross the flat region.

In summary, the first three choices are correct.</p></details>


### **References**: 

<details><summary>CLICK ME</summary><p>- ["Gradient Descent With Momentum from Scratch"](https://machinelearningmastery.com/gradient-descent-with-momentum-from-scratch/) covers this question very well and includes practical examples of how to implement momentum.
- ["Momentum"](https://en.wikipedia.org/wiki/Stochastic_gradient_descent#Momentum) is the Wikipedia page covering momentum as part of [Stochastic Gradient Descent](https://en.wikipedia.org/wiki/Stochastic_gradient_descent).
- The ["Deep Learning"](https://amzn.to/3CSjPkR) book by Goodfellow, et. al. is a fantastic source covering this topic.</p></details>

-----------------------

## Date - 2022-05-29


## Title - Choosing the wrong metric


### **Question** :

Let's assume you are working with a severely imbalanced dataset. 

We've all been there. It's a pretty typical scenario.

Now let's imagine you want to split the data into two categories using a classification learning algorithm.

It's hard to pick the best evaluation metric for this problem if we don't know what we want to accomplish. But at least we can rule out the ones that we shouldn't use.

**Which of the following metrics should you avoid using when evaluating your model's performance?**


### **Choices** :

- Recall
- Precision
- F1-Score
- Accuracy


### **Answer** :

<details><summary>CLICK ME</summary><p>0001</p></details>


### **Explaination** :

<details><summary>CLICK ME</summary><p>Let's illustrate this with a hypothetical example. 

Let's imagine that your team wants to build a machine learning model to predict whether a specific car will get in an accident. 

You are pretty funny, so you decide to play a prank on everyone else by committing this as a solution to the problem:

```
def is_the_car_going_to_crash_today():
    return False
```

Your team evaluates the model against a test set, and your dummy code is 99% accurate!

The National Safety Council reports that the odds of being in a car crash in the United States are less than 1%. This means that even the dumb function above will be very accurate!

The problem here is probably obvious by now: Accuracy is not a good metric when you face a very imbalanced problem. You can achieve very high accuracy even with a model that does nothing useful.

Some examples of imbalanced problems:
* Detecting fraudulent transactions
* Classifying spam messages
* Determining if a patient has cancer

The other three metrics will give you much more information than accuracy, depending on the problem and how you want to approach it.

In summary, the fourth choice is the correct answer to this question.</p></details>


### **References**: 

<details><summary>CLICK ME</summary><p>* [Random Oversampling and Undersampling for Imbalanced Classification](https://machinelearningmastery.com/random-oversampling-and-undersampling-for-imbalanced-classification/)
* Check ["Failure of Classification Accuracy for Imbalanced Class Distributions"](https://machinelearningmastery.com/failure-of-accuracy-for-imbalanced-class-distributions/) to understand why accuracy fails when working with imbalanced datasets.
* If you are into Twitter, [here is a much more detailed story](https://twitter.com/svpino/status/1357302018428256258) about predicting crashes with 99% accuracy.</p></details>

-----------------------

