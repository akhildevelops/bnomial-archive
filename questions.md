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

## Date - 2022-05-30


## Title - A non-boring government job


### **Question** :

Nobody thought that Hazel's job was going to be this interesting.

Her first assignment working for the government was in a lab, but not any lab. She was working with bright people on audio surveillance applications.

A week in and she got a package with maximum urgency. The government recorded a known terrorist at a coffee shop, but unfortunately, the audio was almost inaudible because the music was playing simultaneously.

Hazel needs to clean the audio so they can listen to the target.

**Which of the following techniques should Hazel use?**


### **Choices** :

- Hazel could use Independent Component Analysis to reveal the mixed-signal sources from the audio recording.
- Hazel could use a clustering algorithm to cluster the voice and the music apart on the audio recording.
- Hazel could use supervised learning to identify the signal coming from the voice from the signal coming from the music.
- There's not a good way to clean the audio.


### **Answer** :

<details><summary>CLICK ME</summary><p>1000</p></details>


### **Explaination** :

<details><summary>CLICK ME</summary><p>This problem is known as the "Cocktail Party Problem," where we need to separate two independent but previously mixed audio sources. If we make some assumptions, Hazel should be able to solve this problem.

Let's go straight to the correct answer: [Independent Component Analysis](https://en.wikipedia.org/wiki/Independent_component_analysis) (ICA) is a dimensionality reduction algorithm that should do the trick for Hazel. Here is a quote and an image depicting the problem from ["A Tutorial on Independent Component Analysis"](https://arxiv.org/pdf/1404.2986.pdf):

> Solving blind source separation using ICA has two related interpretations – filtering and dimensional reduction. (...) Filtering data based on ICA has found many applications (...), most notably audio signal processing.

![ICA](https://user-images.githubusercontent.com/1126730/169366606-b019ba62-7999-45a1-8c37-abe333e8f932.png)

Keep in mind that, for ICA to work, the source signals have to be independent of each other and not normally distributed. 

Using a clustering algorithm to separate both signals sounds like something that could potentially work, even if for a constrained use case. I wouldn't be surprised if clustering has been used before for this purpose, but I couldn't find any successful examples of solving the Cocktail Party Problem using clustering, so this is not a correct answer for this question.

The third choice argues for a supervised algorithm. I can see a scenario where we could model the problem in a way that a dataset and a trained neural network help us separate the sources, but this doesn't seem like a viable alternative for this case. I didn't find any examples about this either, so this choice is also incorrect.

In summary, the first choice is the correct answer to this question.</p></details>


### **References**: 

<details><summary>CLICK ME</summary><p>* [A Tutorial on Independent Component Analysis](https://arxiv.org/pdf/1404.2986.pdf)
* [Cocktail Party Problem - Eigentheory and Blind Source Separation Using ICA](https://gowrishankar.info/blog/cocktail-party-problem-eigentheory-and-blind-source-separation-using-ica/)
* [Independent component analysis](https://en.wikipedia.org/wiki/Independent_component_analysis)</p></details>

-----------------------

## Date - 2022-05-31


## Title - Fewer false negatives


### **Question** :

Allison is the Chief Data Scientist of the hospital, and she's been leading a revolutionary machine learning application to help identify patients that could potentially develop a rare disease.

The main goal of the model is to identify every patient that's prone to developing the condition. Allison has worked very hard to reduce the number of false negatives as much as possible.

**From the following list, select every accurate statement describing Allison's situation.**


### **Choices** :

- Allison has worked hard to ensure her model has high sensitivity.
- Sensitivity is the same as the True Positive Rate of the model.
- Higher sensitivity means that the model minimizes the number of false negatives.
- Allison can compute the True Positive Rate of her model by dividing the number of patients that could develop the disease by all patients selected as positive by the model.


### **Answer** :

<details><summary>CLICK ME</summary><p>1111</p></details>


### **Explaination** :

<details><summary>CLICK ME</summary><p>The problem gives us an important clue about what Allison has been doing: she wants to reduce the number of false negatives as much as possible.

Let's start from the beginning and work on this problem step by step. 

A positive sample represents a patient that could develop the disease, and a negative sample represents a patient that will not develop it. Allison wants to reduce the number of false negatives, which is the number of patients that could become sick, but the model misses. In other words, if a patient could become ill and the model misses it, the hospital won't be able to offer treatment, so Allison wants to make sure that happens as infrequently as possible.

Sensitivity refers to the probability of selecting a patient as positive if the person has a genuine chance of developing the disease. We can compute sensitivity by dividing every true positive patient by every patient we think is positive. In short, `sensitivity = TP / P`. 

A model with high sensitivity minimizes the number of false negatives. To get here, we can look back at the formula for sensitivity and break down positive samples (P) into True Positives (TP) + False Negatives (FN). This will give us that `sensitivity = TP / (TP + FN)`. The more False Negatives we have, the lower the sensitivity, so a high-sensitive model keeps false negatives as low as possible.

Allison wants to keep the number of false negatives down, so she wants a high-sensitive model. This makes the first and third choices correct answers to this question.

Finally, we can compute the model's [True Positive Rate](https://www.ibm.com/docs/en/cloud-paks/cp-data/4.0?topic=overview-true-positive-rate-tpr) (TPR) as `TPR = TP / P`. Notice how this is the same formula we use to calculate sensitivity, so we can conclude that the second choice is also correct. The fourth choice's description of True Positive Rate is conveniently also accurate.

In summary, every single choice of this question is correct.</p></details>


### **References**: 

<details><summary>CLICK ME</summary><p>* Everything you need to know about [Sensitivity and specificity](https://en.wikipedia.org/wiki/Sensitivity_and_specificity) you can find in Wikipedia.
* ["Machine Learning – Sensitivity vs Specificity Difference"](https://vitalflux.com/ml-metrics-sensitivity-vs-specificity-difference/) is a great article covering the differences between these two concepts.</p></details>

-----------------------

## Date - 2022-06-01


## Title - The Fukushima nuclear disaster


### **Question** :

The Fukushima nuclear disaster was the most severe nuclear accident since Chernobyl. Together, they have been the only ones with a level 7 classification on the International Nuclear and Radiological Event Scale.

In 2011, an earthquake followed by a tsunami caused the disaster in the Japanese plant, and it all traces back to a mistake in the safety model.

The engineers used historical earthquake data to build a regression model to determine the likelihood of significant earthquakes. Instead of using the accepted [Gutenberg-Richter](https://en.m.wikipedia.org/wiki/Gutenberg%E2%80%93Richter_law) model, they saw a kink in the data and assumed the appropriate regression was not linear but polynomial.

The correct linear model would have predicted that earthquakes of 9.0 magnitude were 70 times more likely than what the incorrect polynomial model predicted. But the engineers, in their pursuit of following the data too closely, came up with a very different conclusion.

The plant was designed to withstand a maximum earthquake of 8.6 magnitude and a tsunami as high as 5.7 meters. The earthquake of 2011 measured 9.0 and resulted in a 14-meter high tsunami.

**How would you summarize the mistake made by the engineers in this incident?**


### **Choices** :

- The engineers built a model that wasn't powerful enough and ended up underfitting the historical earthquake data.
- The engineers built a model that wasn't powerful enough and ended up overfitting the historical earthquake data.
- The engineers built a model that was too powerful and ended up overfitting the historical earthquake data.
- The engineers built a model that was too powerful and ended up underfitting the historical earthquake data.


### **Answer** :

<details><summary>CLICK ME</summary><p>0010</p></details>


### **Explaination** :

<details><summary>CLICK ME</summary><p>When designing the model, the engineers saw a kink in the data. A linear model couldn't follow those data points closely, so they switched to a more complex, polynomial model.

The most important result of the [Gutenberg-Richter law](https://en.m.wikipedia.org/wiki/Gutenberg%E2%80%93Richter_law) is that the relationship between the magnitude of an earthquake and the logarithm of the probability that it happens is linear. The engineers ignored this.

This is a devastating example of overfitting. Here is an excerpt from [Berkeley's machine learning crash course](https://ml.berkeley.edu/blog/posts/crash-course/part-4/):

> As the name implies, overfitting is when we train a predictive model that "hugs" the training data too closely. In this case, the engineers knew the relationship should have been a straight line, but they used a more complex model than they needed to.

The third choice is the correct answer to this question.</p></details>


### **References**: 

<details><summary>CLICK ME</summary><p>* [Fukushima: The Failure of Predictive Models](https://mpra.ub.uni-muenchen.de/69383/1/MPRA_paper_69383.pdf)
* [Machine Learning Crash Course: Part 4](https://ml.berkeley.edu/blog/posts/crash-course/part-4/)
* [Gutenberg–Richter law](https://en.m.wikipedia.org/wiki/Gutenberg%E2%80%93Richter_law)</p></details>

-----------------------

## Date - 2022-06-02


## Title - Classifying waste


### **Question** :

A group of students decided to build an application to classify household waste using smartphone pictures.

They want to start with a simple solution, so they are focusing the first version on the most commonly found types of waste: liquid, solid, organic, recyclable, and hazardous waste.

The tricky part of the application is that it needs to recognize every type of waste present on every image uploaded by users.

The students decided to use a convolutional neural network to solve this problem. The only question left is on the best way to architect it.

**Which of the following would be the best approach to design this network?**


### **Choices** :

- The output layer of the network should have a softmax activation function. The loss function should be categorical cross-entropy.
- The output layer of the network should have a sigmoid activation function. The loss function should be binary cross-entropy.
- The output layer of the network should have a softmax activation function. The loss function should be binary cross-entropy.
- The output layer of the network should have a sigmoid activation function. The loss function should be categorical cross-entropy.


### **Answer** :

<details><summary>CLICK ME</summary><p>0100</p></details>


### **Explaination** :

<details><summary>CLICK ME</summary><p>The students are trying to build a [multi-label classification](https://en.wikipedia.org/wiki/Multi-label_classification) model. In multi-label classification, every image might show multiple types of waste. This is different from [multi-class classification](https://en.wikipedia.org/wiki/Multiclass_classification), where a photo would show only one kind of waste.

When building multi-label classification models, we need an output layer where every class is independent. Remember that we can have more than one active class for each input. The softmax activation function doesn't work because it uses every score to output the probabilities of each class. Softmax is the correct output for multi-class classification but not for multi-label classification problems. 

Since we shouldn't use softmax, the first and third choices are incorrect. The sigmoid function converts output scores to a value between 0 and 1, independently of all the other scores.

Multi-label classification problems borrow the same principles from binary classification problems. The difference is that we end up with multiple sigmoid outputs instead of a single one. In our example problem, we have a combination of five different binary classifiers. This is why we should use a binary cross-entropy as the loss function.

In summary, multi-class classification models should use a softmax output with the categorical cross-entropy loss function. Multi-label classification models should use a sigmoid output and the binary cross-entropy loss function.

The second choice is the correct answer to this question.</p></details>


### **References**: 

<details><summary>CLICK ME</summary><p>* The Wikipedia explanation of [Multi-label classification](https://en.wikipedia.org/wiki/Multi-label_classification) should give you most of what you need to understand for this type of task.
* ["Difference between multi-label classification and multi-class classification"](https://towardsdatascience.com/multi-label-image-classification-with-neural-network-keras-ddc1ab1afede) is an excellent article comparing these two types of problems.
* ["How to choose cross-entropy loss function in Keras?"](https://androidkt.com/choose-cross-entropy-loss-function-in-keras/) explains the differences between the loss functions that we discussed in this question.</p></details>

-----------------------

## Date - 2022-06-03


## Title - Everyone on the same page


### **Question** :

Willow overheard her two friends arguing about the best way to handle a few categorical features on their dataset.

One suggested Label encoding, while the other was pushing for One-Hot encoding. Both are popular encoding techniques, but Willow didn't know enough to understand the difference.

She decided to write a quick summary of both techniques to get everyone on the same page, but the discussion had her confused. She came up with two different explanations for each method, but she wasn't sure which one was correct. 

**Which of the following statements are correct about these two encoding techniques?**


### **Choices** :

- One-Hot encoding replaces each label from the categorical feature with a unique integer based on alphabetical ordering.
- One-Hot encoding creates additional features based on the number of unique values in the categorical feature.
- Label encoding replaces each label from the categorical feature with a unique integer based on alphabetical ordering.
- Label encoding creates additional features based on the number of unique values in the categorical feature.


### **Answer** :

<details><summary>CLICK ME</summary><p>0110</p></details>


### **Explaination** :

<details><summary>CLICK ME</summary><p>Before analyzing this question, we need to understand what "categorical data" means.

Categorical data are variables that contain label values rather than numeric values. For example, a variable representing the weather with values "sunny," "cloudy," and "rainy" is a categorical variable.

Although some algorithms can use categorical data directly, the majority can't: they require the data to be numeric. We can use One-Hot or Label encoding to do this.

[One-Hot encoding](https://hackernoon.com/what-is-one-hot-encoding-why-and-when-do-you-have-to-use-it-e3c6186d008f) creates a new feature for each unique value of the original categorical variable.

For example, assume we have a dataset with a single feature called "weather" that could have the values "sunny," "cloudy," and "rainy." Applying One-Hot Encoding will get us a new dataset with three features, one for each value of the original "weather" column. 

A sample that had the value "cloudy" in the previous column will now have the value 0 for both "sunny" and "rainy" and the value 1 under the "cloudy" feature.

This means that the second choice is the correct explanation of how One-Hot Encoding works.

On the other hand, [Label encoding](https://www.mygreatlearning.com/blog/label-encoding-in-python/) replaces each categorical value with a consecutive number starting from 0. 

For example, Label Encoding would replace our weather feature with a new one containing the values 0 instead of "cloudy," 1 instead of "rainy," and 2 instead of "sunny."

This means that the third choice is the correct explanation of how Label Encoding works.

Therefore, the second and third choices are the correct answers to this question.</p></details>


### **References**: 

<details><summary>CLICK ME</summary><p>* ["What is One Hot Encoding? Why and When Do You Have to Use it?"](https://hackernoon.com/what-is-one-hot-encoding-why-and-when-do-you-have-to-use-it-e3c6186d008f) is an excellent introduction to One-Hot encoding.
* ["Label Encoding in Python Explained"](https://www.mygreatlearning.com/blog/label-encoding-in-python/) is an introduction to Label encoding.
* ["One-Hot Encoding vs. Label Encoding using Scikit-Learn"](https://www.analyticsvidhya.com/blog/2020/03/one-hot-encoding-vs-label-encoding-using-scikit-learn/) covers both techniques and when to use each one.</p></details>

-----------------------

## Date - 2022-06-04


## Title - Cutting down neurons


### **Question** :

Denise had an idea she wanted to try on the neural network she built to identify handwritten digits.

Her output layer had 10 neurons, one for each digit she wanted to
recognize. She thought she could optimize the training process by cutting the number of neurons down to 4.

The goal of the model was to recognize the digit represented by an input image, and with 4 neurons, she could encode a total 16 different values, so she thought this was enough.

After training for some time, Denise found out that the network didn't perform well.

**What conclusions can you draw from Denise's experience?**


### **Choices** :

- To get better results, Denise should experiment with different optimization algorithms and learning rate values.
- Denise's network is working correctly. She can improve the results by modifying her evaluation criteria and discarding any mistakes due to the extra capacity supported by her new architecture.
- With this architecture, the first output neuron has to decide what the most significant bit of the digit represented by the image was. Unfortunately, there's no apparent relationship between the shapes that make up a digit and this information.
- Instead of replacing the 10-neuron layer, Denise should have kept it as a hidden layer and added the 4-neuron layer as the new output. The new network should be capable of finding the bitwise representation of the digit without much trouble.


### **Answer** :

<details><summary>CLICK ME</summary><p>0011</p></details>


### **Explaination** :

<details><summary>CLICK ME</summary><p>In his excellent book [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/chap1.html#a_simple_network_to_classify_handwritten_digits), [Michael Nielsen](https://twitter.com/michael_nielsen) presents this problem and his results after trying both architectures:

> The ultimate justification is empirical: we can try out both network designs, and it turns out that, for this particular problem, the network with 10 output neurons learns to recognize digits better than the network with 4 output neurons. 

But why does the 10-neuron output network outperforms the 4-neuron output network?

> If we had 4 outputs, then the first output neuron would be trying to decide what the most significant bit of the digit was. And there's no easy way to relate that most significant bit to simple shapes (...) It's hard to imagine that there's any good historical reason the component shapes of the digit will be closely related to (say) the most significant bit in the output.

This means that the third choice is a correct explanation of what's happening to Denise with her new architecture. She won't solve the problem by exploring alternate optimization functions or experimenting with different learning rate values. Her architecture is fundamentally flawed.

The fourth choice is a potential avenue that Denise could use to improve the results. She could keep the original layer with 10 neurons to identify the correct digit but add the extra 4-neuron layer as the output to find the digit's binary representation. The network should have no issues solving this problem.

In summary, the third and fourth choices are the correct answer.</p></details>


### **References**: 

<details><summary>CLICK ME</summary><p>* ["Using neural nets to recognize handwritten digits"](http://neuralnetworksanddeeplearning.com/chap1.html#a_simple_network_to_classify_handwritten_digits) is Michael Nielsen's book chapter that discusses this problem.
* ["But what is a neural network?"](https://www.youtube.com/watch?v=aircAruvnKk) is a YouTube video with one of the best explanations out there about how neural networks work.</p></details>

-----------------------

## Date - 2022-06-05


## Title - Regression x 4


### **Question** :

Here are four popular machine learning methods.

Imagine you want to build a simple binary classification model. Your goal is to predict whether a sample is positive or negative.

You could make any of these four algorithms give you the results you want with enough work. That's awesome, but you are interested in the easiest way to make this happen.

**Which of these four algorithms would you use?**


### **Choices** :

- Linear regression
- Lasso regression
- Logistic regression
- Random Forest regression


### **Answer** :

<details><summary>CLICK ME</summary><p>0010</p></details>


### **Explaination** :

<details><summary>CLICK ME</summary><p>The correct answer is Logistic regression. 

[Logistic regression](https://en.wikipedia.org/wiki/Logistic_regression) is an excellent fit for binary classification tasks. It outputs the probability of one event, in our case, the probability of a sample being positive.

All other methods are used to perform regression, in which the algorithm will predict a continuous outcome. We, however, want a categorical output, so logistic regression is the best approach.</p></details>


### **References**: 

<details><summary>CLICK ME</summary><p>- Check out ["Logistic Regression for Machine Learning"](https://machinelearningmastery.com/logistic-regression-for-machine-learning/) for an introduction to Logistic regression.
- ["Logistic Regression"](https://en.wikipedia.org/wiki/Logistic_regression) is the Wikipedia page introducing logistic regression.
- For a list and a quick introduction to regression algorithms, check out ["5 Regression Algorithms you should know – Introductory Guide!"](https://www.analyticsvidhya.com/blog/2021/05/5-regression-algorithms-you-should-know-introductory-guide/)</p></details>

-----------------------

## Date - 2022-06-06


## Title - Supervised learning workshop


### **Question** :

Lydia is going to be teaching a new machine learning class.

Among other things, she will be covering Supervised Learning techniques. Lydia knows how important this is, so she is preparing to turn the class into a giant hands-on workshop.

The University has access to many different datasets, and Lydia decides to pick a few interesting problems with enough data for students to explore.

**Which of the following problems should Lydia pick to teach supervised learning?**


### **Choices** :

- Determine whether a website displays content for a mature audience.
- Learn the best way to split a group of car buyers into different categories based on their buying patterns.
- Given the medical records from patients suffering a specific illness, learn whether we can split them into different groups for better treatment.
- Predict next year's crop yields taking into account data of the past decade.


### **Answer** :

<details><summary>CLICK ME</summary><p>1001</p></details>


### **Explaination** :

<details><summary>CLICK ME</summary><p>There are always multiple ways to approach these problems, but some are better for [supervised learning](https://en.wikipedia.org/wiki/Supervised_learning), while others can benefit from [unsupervised learning](https://en.wikipedia.org/wiki/Unsupervised_learning).

To answer this question, we need to assume that the datasets from the university are labeled when necessary. Remember that supervised learning algorithms require these labels.

Let's start with the first choice. Given any website, we want a "Yes" or "No" answer depending on whether the site displays mature content. Lydia could tackle this problem with a binary classification algorithm, which is a supervised learning technique. 

Splitting a group of car buyers into different categories requires a technique that helps Lydia cluster buyers based on their buying patterns. We can't foresee these patterns beforehand, so any problem that involves finding out the best way of grouping samples is a good fit for clustering algorithms. [Clustering](https://en.wikipedia.org/wiki/Cluster_analysis) is an unsupervised learning technique, so this option is not a good fit for supervised learning. 

The same happens with the problem related to the medical records. We don't have a predefined set of categories to split the group of patients, so this problem seems to be more amenable to clustering techniques.

Finally, predicting next year's crop yields seems a good candidate for a regression algorithm. Regression algorithms are supervised learning techniques that help us predict a continuous value—in this case, how much a crop will yield. This is another valid answer to this question.

In summary, the first and fourth choices are the correct answers.</p></details>


### **References**: 

<details><summary>CLICK ME</summary><p>* ["Supervised and Unsupervised Machine Learning Algorithms"](https://machinelearningmastery.com/supervised-and-unsupervised-machine-learning-algorithms/) is an excellent introduction to the differences between supervised and unsupervised learning.
* Check out ["Customer Segmentation with Machine Learning"](https://towardsdatascience.com/customer-segmentation-with-machine-learning-a0ac8c3d4d84) for more information about how to tackle problems where we need to cluster samples into groups.</p></details>

-----------------------

## Date - 2022-06-07


## Title - Making email fun


### **Question** :

Let's be honest: dealing with email is not fun.

Waking up to an inbox full of unsolicited messages is the worst way to kick off your day. Email applications do their best, but a lot of spam still gets through the cracks.

How about building your own personalized spam detection model? 

One morning, Sue decided to do it, and after a few iterations, she ended with a working model.

Time to find out how good it is!

**Which method should Sue use to evaluate her spam detection model?**


### **Choices** :

- Sue should use the model's accuracy as defined by the percentage of legitimate messages that go through with respect to the total number of received emails.
- Sue should use the model's recall as defined by the percentage of detected spam messages with respect to the total of spam messages received.
- Sue should use the Fβ-Score of the model with a high value of β.
- Sue should use the Fβ-Score of the model with a low value of β.


### **Answer** :

<details><summary>CLICK ME</summary><p>0001</p></details>


### **Explaination** :

<details><summary>CLICK ME</summary><p>Spam detection is an imbalanced problem: you will always have more legitimate emails than spam emails. 

Whenever you need to work with an imbalanced dataset, accuracy will not be a good metric to decide how good your model is. You can achieve very high accuracy even with a model that does nothing useful. For example, if only 1% of the emails you receive are spam, by simply assuming none of it is spam, your model will be 99% accurate. Therefore, the first choice is not a good approach for Sue.

The recall is a helpful metric to understand how much spam you can detect, but by itself could also be deceiving. For example, Sue's model could flag every single email message as spam, which will give her a 100% recall. This, of course, it's not helpful, so the second choice is also not correct.

Using the [Fβ-Score](https://en.wikipedia.org/wiki/F-score), however, is a good choice. 

The Fβ score lets us combine precision and recall into a single metric. When using β = 1, we place equal weight on precision and recall. For values of β > 1, recall is weighted higher than precision; for values of β < 1, precision is weighted higher than recall. 

You are probably familiar with F1-Score. F1-Score is just Fβ-Score with β = 1.

Sue doesn't want to flag valid legitimate messages as spam. Doing this runs the risk of people missing important emails. Therefore, a good strategy is to prioritize a system with high precision, so using a lower value of β is the way to go. Therefore, the correct answer is the fourth choice.

Notice that by prioritizing the precision of her model, Sue will let some spam messages through. Although this is not ideal, it's a better outcome than getting the spam filter to catch legitimate emails.</p></details>


### **References**: 

<details><summary>CLICK ME</summary><p>* ["What is the F-Score?](https://deepai.org/machine-learning-glossary-and-terms/f-score) is a short introduction to this metric.
* For a more in-depth analysis of the Fβ-Score, check ["A Gentle Introduction to the Fbeta-Measure for Machine Learning"](https://machinelearningmastery.com/fbeta-measure-for-machine-learning).
* Check ["Failure of Classification Accuracy for Imbalanced Class Distributions"](https://machinelearningmastery.com/failure-of-accuracy-for-imbalanced-class-distributions/) to understand why accuracy fails when working with imbalanced datasets.</p></details>

-----------------------

## Date - 2022-06-08


## Title - Non-linearities


### **Question** :

River learned an important lesson when trying to implement a neural network from scratch: 

For her network to learn anything useful, she needed to introduce non-linearities.

Whenever she didn't do it, the results were utter trash.

**Which of the following will add non-linearities to River's neural network?**


### **Choices** :

- Using Rectifier Linear Unit (ReLU) as an activation function.
- Adding convolution operations to the network.
- Using Stochastic Gradient Descent to train the network.
- Implementing the backpropagation process.


### **Answer** :

<details><summary>CLICK ME</summary><p>1000</p></details>


### **Explaination** :

<details><summary>CLICK ME</summary><p>For a neural network to learn complex patterns, we need to ensure that the network can approximate any function, not only linear ones. This is why we call it "non-linearities."

The way we do this is by using activation functions. 

An interesting fact: the [Universal approximation theorem](https://en.wikipedia.org/wiki/Universal_approximation_theorem) states that, when using non-linear activation functions, we can turn a two-layer neural network into a universal function approximator. This is an excellent illustration of how powerful neural networks are.

Some of the most popular activation functions are [sigmoid](https://en.wikipedia.org/wiki/Logistic_function), and [ReLU](https://machinelearningmastery.com/rectified-linear-activation-function-for-deep-learning-neural-networks). Therefore, the first choice is the correct answer to this question.

The second choice is incorrect; a [convolution operation is a linear operation](https://en.wikipedia.org/wiki/Convolution#Properties). You can check [this answer](https://ai.stackexchange.com/questions/19879/arent-all-discrete-convolutions-not-just-2d-linear-transforms) in Stack Exchange for an excellent explanation.

Finally, neither Stochastic Gradient Descent nor backpropagation has anything to do with the linearity of the network operations. Therefore, they aren't correct answers.</p></details>


### **References**: 

<details><summary>CLICK ME</summary><p>* Check ["Activation function"](https://en.wikipedia.org/wiki/Activation_function) from Wikipedia to understand more about this topic.
* I find the ["Universal approximation theorem"](https://en.wikipedia.org/wiki/Universal_approximation_theorem) fascinating.</p></details>

-----------------------

## Date - 2022-06-09


## Title - Accuracy as a loss function


### **Question** :

Luna knows that the entire goal of gradient descent is to minimize the value of a function. The lower its value, the better the model will be. She has been thinking about designing a custom loss function for her use case.

She wants to use the inverse of the model's accuracy as the loss function. This way, the model will try to minimize the number of mistakes it makes by looking directly at the accuracy of the predictions.

Unfortunately, she soon discovers that this doesn't work. 

**What is the problem with this approach?**


### **Choices** :

- Accuracy is not a differentiable function, so it can't be optimized using gradient descent.
- Luna wants to optimize for high accuracy but gradient descent is a minimization algorithm; the opposite of what she needs.
- Minimizing the inverse of the accuracy is a very slow process because of the extra computations needed to compute the final value.
- Gradient descent only works with a predefined set of loss functions.


### **Answer** :

<details><summary>CLICK ME</summary><p>1000</p></details>


### **Explaination** :

<details><summary>CLICK ME</summary><p>We use loss functions to optimize a model. We use accuracy to measure the performance of that model.

Usually, we can see how the accuracy of a classification model increases as the loss decreases. This is not always the case, however. The loss and the accuracy measure two different aspects of a model. Two models with the same accuracy may have different losses. 

An important insight: The loss function must be continuous, but accuracy is discrete. When training a neural network with gradient descent, we need a differentiable function because the algorithm can't optimize non-differentiable functions. One of the required characteristics for a function to be differentiable is that it must be continuous. Since accuracy isn't, we can't use it.

This makes the first choice the correct answer to this question.

The second and third choices assume that we can use accuracy as the loss function one way or the other, so they are incorrect. The fourth choice claims that gradient descent can only work with a predefined subset of functions, which is also wrong.</p></details>


### **References**: 

<details><summary>CLICK ME</summary><p>- ["Loss and Loss Functions for Training Deep Learning Neural Networks"](https://machinelearningmastery.com/loss-and-loss-functions-for-training-deep-learning-neural-networks/) is a great introduction to loss functions.
- Check ["How to Choose Loss Functions When Training Deep Learning Neural Networks"](https://machinelearningmastery.com/how-to-choose-loss-functions-when-training-deep-learning-neural-networks/) for a guide on how to choose and implement different loss functions.</p></details>

-----------------------

## Date - 2022-06-10


## Title - Psychologists speak another language


### **Question** :

A group of psychologists is visiting the office, and Scarlett is in charge of showing them around.

The first stop will be in the Data Science department. They are very excited about showing them the results of their latest machine learning model.

Ten minutes into the presentation, it's painfully apparent that the crew is not fully grasping what is going on. Scarlett decides to summarize her ideas using a familiar language: statistics.

In statistics, the notion of statistical error is an integral part of hypothesis testing. There are two types of errors when testing the null hypothesis: type I and type II errors. Scarlett wants to explain their results regarding the latter.

**Do you remember what the correct definition of a type II error is?**


### **Choices** :

- A type II error occurs when the null hypothesis is true and is not rejected.
- A type II error occurs when the null hypothesis is true but is rejected.
- A type II error occurs when the null hypothesis is false but is not rejected.
- A type II error occurs when the null hypothesis is false and is rejected.


### **Answer** :

<details><summary>CLICK ME</summary><p>0010</p></details>


### **Explaination** :

<details><summary>CLICK ME</summary><p>It makes sense for those who are more used to machine learning terminology to compare type I and type II errors with false positives and false negatives.

Type I errors are the same as false positives. For example, if we mark a valid email as spam, we are in the presence of a false positive. Type I errors are the rejection of a true [null hypothesis](https://www.investopedia.com/terms/n/null_hypothesis.asp) by mistake.

Type II errors are the same as false negatives. For example, if we let a spam message pass as a valid email, we are in the presence of a false negative. This is a type II error because we accept the conclusion of the email being good, even though it is incorrect. Type II errors are the acceptance of a false null hypothesis by mistake.

In other words, a type II error is when we incorrectly accept the null hypothesis even though the alternative hypothesis is true. Therefore, The third choice is the correct answer to this question.</p></details>


### **References**: 

<details><summary>CLICK ME</summary><p>* ["What Is a Null Hypothesis?"](https://www.investopedia.com/terms/n/null_hypothesis.asp) covers the basics you need to understand before going into hypothesis testing.
* Check out ["Type I and type II errors"](https://en.wikipedia.org/wiki/Type_I_and_type_II_errors) for the definition and examples of each type of error. 
* ["Understanding Null Hypothesis Testing"](https://opentextbc.ca/researchmethods/chapter/understanding-null-hypothesis-testing/) is an excellent article about hypothesis testing.</p></details>

-----------------------

## Date - 2022-06-11


## Title - Penelope is looking for more


### **Question** :

Penelope needs to finish her homework. The final step is to make her neural network "deeper."

She tried to solve the problem using a shallow network architecture, but her professor wanted her to try a deeper network.

**What should Penelope do to turn her architecture into a deep neural network?**


### **Choices** :

- Penelope should increase the dimensionality of the input data.
- Penelope should add more layers to the network.
- Penelope should use images, text, voice, or video as the input.
- Penelope should add more neurons to the existing layers.


### **Answer** :

<details><summary>CLICK ME</summary><p>0100</p></details>


### **Explaination** :

<details><summary>CLICK ME</summary><p>A deep neural network is an artificial neural network with multiple layers between the input and output layers. The more layers we add, the "deeper" the network becomes.

A deep neural network can process all sorts of data. Neither the type of input nor its dimensionality has any relationship with the depth of the network. Therefore, neither the first nor the third choices are correct.

Adding neurons alone doesn't change the depth of the network either. It does change its capacity, but a network with the same number of layers but more neurons will still have the same depth. 

The only choice for Penelope is to add more layers to the network.</p></details>


### **References**: 

<details><summary>CLICK ME</summary><p>* Wikipedia's definition of ["Deep neural networks"](https://en.wikipedia.org/wiki/Deep_learning#Deep_neural_networks) serves as a succinct summary of what deep neural networks are.
* Check ["A Layman’s Guide to Deep Neural Networks"](https://towardsdatascience.com/a-laymans-guide-to-deep-neural-networks-ddcea24847fb) for a non-mathematical introduction to deep neural networks.</p></details>

-----------------------

## Date - 2022-06-12


## Title - Low-variance model


### **Question** :

Usually, the best approach is to start experimenting with a few different algorithms to narrow down the possibilities and pick a good path forward.

That's what Sophia did. She ran her dataset through four different algorithms and noticed something peculiar.

Since she had a lot of data, she trained each algorithm with different batches separately. Only one of the models gave consistent results regardless of what portion of the data she used.

Sophia knew this was a variance issue. Low-variance models usually produce consistent results regardless of the data used to train them.

**Which of the following algorithms was the one giving consistent results?**


### **Choices** :

- Support Vector Machine (SVM)
- Decision Trees
- Logistic Regression
- k-Nearest Neighbors (KNN)


### **Answer** :

<details><summary>CLICK ME</summary><p>0010</p></details>


### **Explaination** :

<details><summary>CLICK ME</summary><p>Every machine learning algorithm deals with three types of errors: bias, variance, and irreducible error. We need to focus specifically on the variance error to answer this question.

Here is what [Jason Brownlee](https://machinelearningmastery.com/gentle-introduction-to-the-bias-variance-trade-off-in-machine-learning/) has to say about variance: "Variance is the amount that the estimate of the target function will change if different training data was used."

In other words, variance refers to how much the answers given by the model will change if we use different training data. The model has low variance if the answers stay the same when using different portions of our training dataset.

Generally, linear models with little flexibility have low variance. For example, Linear and logistic regression are low-variance models. Nonlinear algorithms with a lot of flexibility have high variance, for example, Decision Trees, Support Vector Machines, and k-Nearest Neighbors.

Therefore, the third choice is the correct answer to this question.</p></details>


### **References**: 

<details><summary>CLICK ME</summary><p>* ["Gentle Introduction to the Bias-Variance Trade-Off in Machine Learning"](https://machinelearningmastery.com/gentle-introduction-to-the-bias-variance-trade-off-in-machine-learning/) is Jason Brownlee's article covering bias, variance, and their tradeoff.
* The Wikipedia page on bias and variance is also a good resource: ["Bias–variance tradeoff"](https://en.wikipedia.org/wiki/Bias–variance_tradeoff).
* In case you like the simplicity of Twitter threads, here is one for you about this topic: ["Bias, variance, and their relationship with machine learning algorithms"](https://twitter.com/svpino/status/1390969728504565761).</p></details>

-----------------------

## Date - 2022-06-13


## Title - Looking for labels


### **Question** :

The company had a lot of data, but none was labeled. 

As soon as the team started planning the work, Blake's first recommendation was to look into Supervised Learning. She knew, however, that without labeled data, they weren't going anywhere.

There are many different ways to produce labels, and Blake will have to decide how to move forward.

**Which of the following techniques could Blake use to label the data?**


### **Choices** :

- Assemble a team of people that go through the data and label each sample.
- Use feedback from an existing process to automatically produce the labels.
- Use a Supervised Learning technique to infer the labels directly from the existing data.
- Use Semi-Supervised Learning to propagate labels across all of your data.


### **Answer** :

<details><summary>CLICK ME</summary><p>1100</p></details>


### **Explaination** :

<details><summary>CLICK ME</summary><p>If Blake wants to use a Supervised Learning method, she has no other option than to produce labels for the data. There are many different techniques she can use to accomplish this.

The most common way to label data is to use human labelers. Blake could assemble a team that will go through each sample assigning the appropriate label. For example, assuming the company wants to classify car pictures, the labelers could review each image and set the correct make and model of the car. Therefore, the first choice is correct.

Sometimes, we can use feedback from an existing process to create labels, also known as "direct labeling." For example, a video site recommending movies to different users can use actual clicks from its audience to determine which posters work and which don't. 

Unfortunately, direct labeling is very dependent on your use case, and it's not something you can always do. Also, notice that direct labeling doesn't necessarily capture the "true ground-truth," but only a useful approximation. Nevertheless, direct labeling is a good approach, so the second choice is correct.

The third choice argues that we could use a Supervised Learning method to infer the labels from the existing dataset, but this doesn't make sense. Supervised Learning requires the existence of labels, and that's what Blake doesn't have. If we could use the data to predict labels, we could also use it to solve the problem in the first place. This choice is incorrect.

Finally, we could use [Semi-Supervised Learning](https://machinelearningmastery.com/semi-supervised-learning-with-label-propagation/) assuming we already have a few labels. For example, if we had 10% of the labels, we could build a model to generate the other 90% of labels. However, there's no indication that Blake has any labeled data, so Semi-Supervised Learning is not an option. This choice is also incorrect.

[Active Learning](https://rapidminer.com/glossary/active-learning-machine-learning/) and [Weak Supervision](https://snorkel.ai/weak-supervision/) are also techniques to generate labels. They aren't part of this question, but it's helpful to know about them.</p></details>


### **References**: 

<details><summary>CLICK ME</summary><p>* The ["Machine Learning Data Lifecycle in Production"](https://www.coursera.org/learn/machine-learning-data-lifecycle-in-production) course in Coursera, part of the [Machine Learning Engineering for Production (MLOps) Specialization](https://www.coursera.org/specializations/machine-learning-engineering-for-production-mlops).
* Check out ["Semi-Supervised Learning With Label Propagation"](https://machinelearningmastery.com/semi-supervised-learning-with-label-propagation/) for an introduction to how to use a few labels with semi-supervised learning.
* ["Active Learning in Machine Learning"](https://rapidminer.com/glossary/active-learning-machine-learning/) is a short explanation of Active Learning, enough if all you need is a high-level overview.
* If you are serious about Active Learning, ["Active Learning Literature Survey"](https://burrsettles.com/pub/settles.activelearning.pdf) is the publication you want to read. 
* ["Weak Supervision: A New Programming Paradigm for Machine Learning"](http://ai.stanford.edu/blog/weak-supervision/) is a good article from Stanford introducing Weak Supervision.
* An introduction to [Weak Supervision](https://snorkel.ai/weak-supervision/) by Snorkel AI, a labeling platform. This one even includes a video.</p></details>

-----------------------

## Date - 2022-06-14


## Title - Rebecca's rotation


### **Question** :

After a very late coffee, Rebecca felt plenty of energy to crack open the book she's been dreading to read the entire week.

It was a dense read. A computer vision masterpiece that went all the into the mathematical reasoning behind every topic.

The latest chapter combined deep learning, linear algebra, and geometric transformations. Rebecca promised herself to watch a movie as soon as she finished the first problem on the topic.

It seemed straightforward: Rebecca had to rotate a two-dimensional square 90 degrees counterclockwise using matrix multiplication. She knew she had to multiply the coordinates of her square with a specific 2x2 matrix, but she didn't remember how exactly it worked.

**Which of the following is the correct matrix R to rotate a 2D square 90 degrees counterclockwise?**


### **Choices** :

- The matrix R is `[[0, -1], [1, 0]]`.
- The matrix R is `[[0, 1], [-1, 0]]`.
- The matrix R is `[[1, 0], [0, 1]]`.
- The matrix R is `[[-1, 0], [0, -1]]`.


### **Answer** :

<details><summary>CLICK ME</summary><p>1000</p></details>


### **Explaination** :

<details><summary>CLICK ME</summary><p>At its core, deep learning mostly boils down to many tensor operations chained together. These operations have a corresponding geometric interpretation, and understanding them is an excellent exercise to shed some light on how deep learning networks work.

When we talk about [rotating](https://en.wikipedia.org/wiki/Rotation_(mathematics)) an object, we can think of moving each point of that object circularly around a center. Assuming that we use a column vector to represent the coordinate of each point, we can use matrix multiplication to rotate the object.

We need a rotation matrix R to multiply with the object's coordinates and obtain the new set of rotated coordinates. The structure of this matrix R to rotate an object counterclockwise is `[[cos(θ), -sin(θ)], [sin(θ), cos(θ)]]` where θ represents the rotation angle.

Rebecca needs to rotate the 2D square 90 degrees. Remember that `cos(90) = 0` and `sin(90) = 1`, so the matrix R that Rebecca needs is `[[0, -1], [1, 0]]`, which is the first choice of this question.

Just for fun, we can go through all the other choices and determine what would be the corresponding rotation angle. 

The second choice rotates the 2D square 90 degrees clockwise—in the opposite direction that Rebecca wanted. Notice how the only difference with the correct answer is the position of the signs.

The third choice does not rotate the square—θ is 0 degrees. To see this, let's start with the matrix `[[cos(θ), -sin(θ)], [sin(θ), cos(θ)]]` and assume we multiply it by a vector `[x, y]` to get a new, rotated vector `[x', y']`:

```
x' = x * cos(θ) - y * sin(θ)
y' = x * sin(θ) + y * cos(θ)
```

Replacing the values of each component as specified in the third choice:

```
x' = x * 1 - y * 0 = x - 0 = x
y' = x * 0 + y * 1 = 0 + y = y
```

Notice how we get the exact coordinates after we apply the rotation. 

Finally, the fourth choice rotates the square 180 degrees counterclockwise. 

In summary, the first choice is the correct answer to this question.</p></details>


### **References**: 

<details><summary>CLICK ME</summary><p>* Check ["Rotations and reflections in two dimensions"](https://en.wikipedia.org/wiki/Rotations_and_reflections_in_two_dimensions) for an explanation of how to rotate and reflect objects in a two-dimensional space.</p></details>

-----------------------

