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

