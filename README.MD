# Data Science Approach
* Obtain: Gather data from relevant sources
* Scrub: Clean data to formats that machine understands
* Explore: Find significant patterns and trends using statistical methods
* Model: Construct models to predict and forecast
* Interpret: Put the results to good use

    * Model is trained. But need to understand strengths and weaknesses
    * Example: Fraud detection or disease dignose requires a model that rarely misses positives and it is okay to have false positive.  


# Statistics
## What is Exploratory Data Analysis? (EDA)
EDA is the process where data scientist visualize, examine, organize and summarize data.

## What are the commom methods of EDA?
* Visualization:
    * Histogram Plots
    * Scatter Plots
    * violine Plots
    * Boxplots

## Statistical sample and populations
* What is population?
A population can be defined as including all people or items with the characteristic one wishes to understand.

* What is sampling?
sampling is the selection of a subset (a statistical sample) of individuals from within a statistical population to estimate characteristics of the whole population

* Why sampling? Advantages of Sampling.
Because there is very rarely enough time or money to gather information from everyone or everything in a population, the goal becomes finding a representative sample (or subset) of that population

* What is probability sampling?
A probability sample is a sample in which every unit in the population has a chance (greater than zero) of being selected in the sample, and this probability can be accurately determined. The combination of these traits makes it possible to produce unbiased estimates of population totals, by weighting sampled units according to their probability of selection.
* Features:
    * Evenry element has a known nonzero probability of being sampled
    * involves random sampling at some point
* Examples: Simple Random Sampling, Systematic Sampling, Stratified Sampling, Probability Proportional to Size Sampling, and Cluster or Multistage Sampling

### Sampling methods
* Simple Random Sampling
* Systematic sampling
* stratified sampling
    * When the population embraces a number of distinct categories, the frame can be organized by these categories into separate "strata." Each stratum is then sampled as an independent sub-population, out of which individual elements can be randomly selected. The ratio of the size of this random selection (or sample) to the size of the population is called a sampling fraction. There are several potential benefits to stratified sampling.
* Probability-proportional-to-size sampling
* Cluster sampling
* Minimax sampling
* Accidental sampling
* Voluntary Sampling
* Line-intercept sampling

### Variables
* Two categories
    * Quantitative/numerical
    * Qualitative/categorical
* Nominal vs. Ordinal
    * Normal scale variable differentiate individual data points
    * Ordinal scale variable can measure direction, size,values, etc.
* Interval vs Ratio
    * Interval ration
    * Ration scales, have inherent zero
* continuous vs. discrete variables


### 


## Data Analysis- Summarizing
* Distribution:
    * Skewness -
        * Right - Tail on right - positive skew
        * Left - Tail on left - negative skew
    * Application: Outliners
* Summarizing Statistics
    * Mean: effected by outliners
    * Mode: 
    * Median: does not affect by outliners
* Measure of Spread Dispersion of one variable -- Variance/Standard Diviation/Coefficient of Variance
    * Population Variance: 
        * ![equation](https://latex.codecogs.com/gif.latex?\sigma&space;^{2}&space;=&space;\frac{1}{N}\sum_{i=1}^{N}(x_{i}&space;-&space;\mu&space;)^{2})
        * Dispersion from mean value
    * Population  Standard Deviation: 
        * ![equation](https://latex.codecogs.com/gif.latex?\sigma&space;=&space;\sqrt{\frac{1}{N}\sum_{i=1}^{N}(x_{i}&space;-&space;\mu&space;)^{2}})
        *  the concentration of the data around the mean of the data set
    * Population Coeffience of Variance (CV)
        * ![equation](https://latex.codecogs.com/gif.latex?CV&space;=&space;\frac{\sigma&space;}{\mu&space;})
    * Sampling Dispersion: N -> n-1
* Measure of Spread Dispersion of Two variables -- Covariance/Covariance Coefficient
    * Covriance of x, y
        * ![equation](https://latex.codecogs.com/gif.latex?Cov(x,&space;y)&space;=&space;\sigma&space;_{xy}&space;=&space;\sum_{i=1}^{N}\frac{(x_{i}-\mu&space;_{x})(y_{i}-\mu&space;_{y})}{N})
        * Weakness:
            * Covariance does not give effective information about the relation between 2 variables as it is not normalized. This means we can’t compare variances over data sets with different scales (like pounds and inches). A weak covariance in one data set may be a strong one in a different data set with different scales. 
    * Corelation
        * Correlation provides a better understanding of the relationship between two variables because it is the Normalized Covariance. 
        * ![equation](https://latex.codecogs.com/gif.latex?Correlation&space;=&space;\rho&space;=&space;\frac{Cov(x,&space;y)}{\sigma&space;_{x}\sigma&space;_{y}})
        * Correlation between two variables DOES NOT mean causation. 
        * Correlation Matrix
        ```python
        corr= df.corr()
        sns.heatmap(corr, cmap=sns.diverging_palette(220,10, as_camp=True))
        ```
        * Pairwise plots
        ```python
        sns.pairplot(df)
        ```
## Normal Distribution and Central Limit Theorem
### Normal distribution
* What is normal distribution?
    * ![equation](https://latex.codecogs.com/gif.latex?Normal&space;Distribution:&space;X&space;~&space;N(\mu&space;,&space;\sigma&space;^{2}))
* What is CLT?
The Central Limit Theorem states that the sampling statistics distribution will look like a normal distribution regardless of the population analyzed
* Application of CLT
Using sampling statistics to estimate population statistics
* Standard Normal Distribution:
    * ![euqation](https://latex.codecogs.com/gif.latex?Standard&space;normal&space;distribution:&space;X\sim&space;N(0,1))
* Transform to stantdard normal distribution:
    * ![equation](https://latex.codecogs.com/gif.latex?X&space;\sim&space;N(\mu&space;,&space;\sigma&space;^{2})&space;\rightarrow&space;Z=\frac{x-\mu&space;}{\sigma&space;}:Z\sim&space;N(0&space;,&space;1))

## Probability
    * Rule of probability:
    for non-mutually exclusive events: ![equation](https://latex.codecogs.com/gif.latex?P(A\cup&space;B)&space;=&space;P(A)&space;&plus;&space;P(B)-P(A\cap&space;B))
    * Permutations: order matters
        * Permutation with repetition: 
        * P without repition: Factorial
        ![equation]P(n,k)= \frac{n!}{(n-k)!}
    * Combinations: order dosent matter
        * Combination with repetition: ![equation](https://latex.codecogs.com/gif.latex?\bar{C}(n,k)=&space;\frac{(n&plus;k-1)!}{(n-1)!k!})
        * Combinatin without repetition: ![equation](C(n,k)= \frac{n!}{(n-k)!k!})
    * Joint Probability
    * Marginal Probability
    * Conditional probability
    * Independence vs. exclusivity
    * Bayes Theorem
    ![equation](https://latex.codecogs.com/gif.latex?P(A\cap&space;B)&space;=&space;P(A|B)P(B)&space;=&space;P(B|A)P(A))
## Hypothesis testing
* What is hypothesis testing?
A statistical hypothesis is an assumption about a population parameter. This assumption may or may not be true. Hypothesis testing refers to the formal procedures used by statisticians to accept or reject statistical hypotheses. The best way to determine whether a statistical hypothesis is true would be to examine the entire population. Since that is often impractical, researchers typically examine a random sample from the population. If sample data are not consistent with the statistical hypothesis, the hypothesis is rejected.
* Null Hypotheis: H0: This describes the existing conditions or present state of affairs. The null hypothesis, denoted by Ho, is usually the hypothesis that sample observations result purely from chance.
* Alternative Hypothesis: Ha: This is used to compare or contrast against the Null Hypothesis.  hypothesis that sample observations are influenced by some non-random cause.
* Hypothesis test steps:
    * State the hypotheses. This involves stating the null and alternative hypotheses. The hypotheses are stated in such a way that they are mutually exclusive. That is, if one is true, the other must be false.
    * Formulate an analysis plan. The analysis plan describes how to use sample data to evaluate the null hypothesis. The evaluation often focuses around a single test statistic.
    * Analyze sample data. Find the value of the test statistic (mean score, proportion, t statistic, z-score, etc.) described in the analysis plan.
    * Interpret results. Apply the decision rule described in the analysis plan. If the value of the test statistic is unlikely, based on the null hypothesis, reject the null hypothesis.
* Decision Errors
    * Type I error. A Type I error occurs when the researcher rejects a null hypothesis when it is true. The probability of committing a Type I error is called the significance level. This probability is also called alpha, and is often denoted by α.
    * Type II error. A Type II error occurs when the researcher fails to reject a null hypothesis that is false. The probability of committing a Type II error is called Beta, and is often denoted by β. The probability of not committing a Type II error is called the Power of the test.
* Decision Rules
    * p value:  The strength of evidence in support of a null hypothesis is measured by the P-value.
        * If the P-value is less than the significance level, we reject the null hypothesis.
    * Region of acceptance
        * The region of acceptance is a range of values. If the test statistic falls within the region of acceptance, the null hypothesis is not rejected. The region of acceptance is defined so that the chance of making a Type I error is equal to the significance level.
    * Region of rejection
        * The set of values outside the region of acceptance is called the region of rejection.
        * If the test statistic falls within the region of rejection, the null hypothesis is rejected.
        *  In such cases, we say that the hypothesis has been rejected at the α level of significance.
* One Tail test
    * A test of a statistical hypothesis, where the region of rejection is on only one side of the sampling distribution, is called a one-tailed test. 
* Two tail test
    * A test of a statistical hypothesis, where the region of rejection is on both sides of the sampling distribution, is called a two-tailed test. 
* z-score and percentiles
    ![equation](https://latex.codecogs.com/gif.latex?z-score:&space;z=\frac{x-\mu&space;}{\sigma&space;})

# Machine Learning
## What is ML models?
Equations that transform input to output
## Types of ML
* Supervised Learning
* Unsupervised learning
* Reinforcement Learning
## Linear Regression -- Cost Functions and Gradient Descent
* Regression: y = f(x) = mx + b
* How to find most appropriate values of m b? -> Least Squared Regression
    * Loss Function: MSE 
    ![equation](https://latex.codecogs.com/gif.latex?MSE(m,&space;b)&space;=&space;\frac{1}{N}\sum_{i=1}^{N}(y_{i}-&space;\hat{y}_{i})^{2})
    * Gradient Descent to minimize cost loss function
## Plolynomial and Multivariate Linear Regression
* ![equation](https://latex.codecogs.com/gif.latex?Polynomial&space;Regression:&space;y&space;=&space;m_{1}x&space;&plus;&space;m_{2}x^{2}&plus;&space;m_{3}x^{3}&space;&plus;...&plus;&plus;&space;m_{n}x^{n}&plus;b)
* ![equation](https://latex.codecogs.com/gif.latex?Multivariate&space;Linear&space;Regression:&space;y&space;=&space;f(x)&space;=&space;w_{1}x_{1}&space;&plus;&space;w_{2}x_{2}&plus;&space;w_{3}x_{3}&space;&plus;...&plus;w_{n}x_{n}&plus;b=\sum_{i=1}^{n}w_{i}x_{i})

## Logistic Regression
### What is Logistic Regression?
In statistics, the logistic model (or logit model) is used to model the probability of a certain class or event existing such as pass/fail, win/lose, alive/dead or healthy/sick. This can be extended to model several classes of events such as determining whether an image contains a cat, dog, lion, etc. Each object being detected in the image would be assigned a probability between 0 and 1, with a sum of one.
### Functions
* Linear Regression: ![equation](https://latex.codecogs.com/gif.latex?f&space;=&space;\mathbf{w}^{\boldsymbol{T}}\boldsymbol{x})
* Logistic regression is the transformation of a linear function with a logistic sigmoid
    * sigmoid: ![equation](https://latex.codecogs.com/gif.latex?g(z)&space;=&space;\frac{1}{1&plus;e^{-z}})
    * Logistic function: ![equation](https://latex.codecogs.com/gif.latex?f(x,w)&space;=&space;\sigma&space;(\boldsymbol{w}^{\boldsymbol{T}}\boldsymbol{x})=\frac{1}{1&plus;e^{-\boldsymbol{w}^{\boldsymbol{T}}\boldsymbol{x}}})
    * Decision Boundary
    * Gradient Descent to minimize functions
### Naive Bayes

### Support Vector Machines (classifier and regressor)
* What is SVM?
 a support-vector machine constructs a hyperplane or set of hyperplanes in a high- or infinite-dimensional space, which can be used for classification, regression, or other tasks like outliers detection. Intuitively, a good separation is achieved by the hyperplane that has the largest distance to the nearest training-data point of any class (so-called functional margin), since in general the larger the margin, the lower the generalization error of the classifier
* Optimal Hyperplane
    * The goal of SVMs is to maximize the margin between the data points of each classes.  The plane or decision boundary is called hyperplane. Points of each class are seperated by hyperplane.
    * Hyperplane can be in multiple dimensions.
* Support Vectors
    * Support Vectors are the points that reside closest to the hyperplane

* SVM Classification model
    * SVMs take output of linear function and for values >1 as one class; values <-1 the other class
    * Non-linear data
        * Uses Kernel to take low dimensional input space to higher dimensional space
* Pros of SVM
    * It works really well with a clear margin of separation
    * It is effective in high dimensional spaces.
    * It is effective in cases where the number of dimensions is greater than the number of samples.
    * It uses a subset of training points in the decision function (called support vectors), so it is also memory efficient.
* Cons of SVM
    * It doesn’t perform well when we have large data set because the required training time is higher
    * It also doesn’t perform very well, when the data set has more noise i.e. target classes are overlapping
    * SVM doesn’t directly provide probability estimates, these are calculated using an expensive five-fold cross-validation. It is included in the related SVC method of Python scikit-learn library.
### Decision Trees 
* Multi-class decision tree, need to find a way to define 'splits'
* Find the root split
* Define and evluate splits -- Gini Gain 
    * Gini Gain (impurity) for the whole dataset
    * probablity of incorrectly classifying a random selected element in the dataset
    * ![equation](https://latex.codecogs.com/gif.latex?G_{initial}&space;=&space;\sum_{i=1}^{C}p_{i}(1-p_{i}))
        * C: number of classes
        * pi: probability of randomly choosing element of class i
    * GLeft: ![equation](https://latex.codecogs.com/gif.latex?G_{Left}&space;=&space;\sum_{i=1}^{C}p_{i}(1-p_{i}))
    * GRight: ![Equation](https://latex.codecogs.com/gif.latex?G_{Right}&space;=&space;\sum_{i=1}^{C}p_{i}(1-p_{i}))
    * Quality of the split = weighting the impurity of each branch by the number of elements it contains
    * GiniGain of the split = Ginitial - Quality of Split that is the total impurity removed with this split
    * Calculate GiniGain for each split and use with the highest GiniGain
    * GiniGain for multiple classes
    * Calculating Gini Gains is how Decision Tree is constructed


### Random Forest
* What is Random Forest?
The output of multiple decision tree classifiers
* Train decision tress on n set of samples and repeat t times
* find the most common output (majority vote) or average (for regression) --> Bagging/Boostrapping

### KNN
* Lazy learner. do not have a training phase.
* Comput classification using training data (can be slow)
* Non-parametric algorithm
* Choosing K
    * k=1 - high variance
    * k ↑ - variance decreased, bias increase
* Distance Metrics
    * Euclidean distance
    * Cosine Distance
* Pros of KNN
    * The algorithm is simple and easy to implement.
    * There’s no need to build a model, tune several parameters, or make additional assumptions.
    * The algorithm is versatile. It can be used for classification, regression, and search (as we will see in the next section).
* Cons of KNN
    * Every time we wish to calculate new input, need to load all training data, then calculate distane to every point. 
    * computing extensively and requires high memory
    * To make KNN work well, need to normalize or scale input data
    * Datasets with a large number of features (i.e. dimensions) will impact KNN performance. To avoid ove

### Model Evaluation Metrics
* Classification -- 
    * Accuracy
    * Confusion Matrix
    * Precision
    * Recall
    * F1-Score: weighted avrage of Precision and Recall
        * Useful in case of class imbalance
    * ROC
        * ROC: Receiver Operating Characteristics
        * goodness of model in distinguishing two classes
        * TPR vs. FPR (Recall vs. 1 - Precision)
    * AUC (Area under the curve)
        * Created using different thresholds in classifier to get TPR and FPR

### Overfitting - Regularization, Generalization, Outliners
* Overfitting vs. Underfitting (Variance vs. bias)
* How to avoid overfitting?
    * Test/Evaluatation
    * Regularization: reduce model complexity
        * ML:
            * L1 and L2 Regularization
            Large weights or gradients manifest as abrupt changes in our model’s decision boundary. By penalizing, we a really making them smaller. 
            * cross validation
        * Deep Learning
            * Early Stopping
            * Drop out
            * Dataset Augmentation


### Neural Networks and Deep learning
#### CNN
* Why CNN needed?
    * Neural Networks don't scale well into image data. For a color image 64px by 64px. Input size of NN will be 64 * 64 * 3 = 12288
    * Image data -> consists of patterns and correlated inputs
* How CNN works?
Input image → Convolution + ReLU → Polling → Convolution + ReLU → Pooling → Fully connected layer → Output layer
* What is convolution?
    * process of combining two functions to produce a third  function
    * third function is called feature map
    * using a filter or kernel that is applied to the input
    * Sliding the filter or kernel over input image
#### RNN
* What is RNN?
    * Feed Forward NN has no concept of time/sequence, each calculation is done only on the current inputs
    * Recurrent Networks take as their input not just the current input, but also what have preceived previously in time.
    * Two inputs: the present input and the past inputs
* Hidden state vector (h)
    * RNN influence not just by weights applied on inputs like a regular NN, but also by a 'hidden state vector (h)' representing the context based on prior inputs/outputs

* Pros of RNN
    * Successful in speech recognition, language modeling, image captioning
* Cons of RNN
    * Memory vanishing gradient and exploading gradient
#### LSTM
* What is LSTM networks?
    * a type of RNN
    * with a memory unit (cell)
    * capable of learning long-term dependencies
#### Feed Forward Propagation
* Universal Approximation Theorem
Mathematically speaking, any neural network architecture aims at finding any mathematical function y= f(x) that can map attributes(x) to output(y). The accuracy of this function i.e. mapping differs depending on the distribution of the dataset and the architecture of the network employed. The function f(x) can be arbitrarily complex. The Univeral Approximation Theorem tells us that Neural Networks has a kind of universality i.e. no matter what f(x) is, there is a network that can approximately approach the result and do the job! This result holds for any number of inputs and outputs.

* Activation Function
    * ![equation](https://latex.codecogs.com/gif.latex?Output&space;=&space;Activiation(w_{i}x_{i}&space;&plus;&space;b))
    * ReLU (Rectified Linear Unit)
        * ![equation](https://latex.codecogs.com/gif.latex?f(x)&space;=&space;max(0,x))
    * Why?
        * Most ML algorithms find non linear data extremely hard to model.
        * advantage of deep learning is the ability to understand nonlinear models
    * Typles of Activation function
        * Sigmoid
        * Hyperbolic Tangent
        * ReLU
        * Radial Basis Function
    * Bias
        * neuron with a trainable constant value
        * shift activation function
* What is Deep?
    * Deep refers to the number of hidden layers
    * Deeper → nonlinear mapping better
    * But could overfit and overly long training time

* Loss function, Gradient Descent, Backpropagation
    * How to find the weights for DL models?
        * with loss function, adjust weights by gredient descent, backpropagation 
    * Type of Loss functions
        * L1
        * L2
        * MSE
        * Cross Entropy ( in binary classification)
        * Hinge Loss
        * MAE
    * Backpropagation
        * determine which direction the weight should move
        * Epoch or Iteration
        * Gradient Descent -- optimization
        * Chain rule
        * Learning Rate
    * Gradient Descent
        * Type of GD
            * Naive Gradient Descent 
                * Compute expensive/slow
                * requires exposure to the entire dataset, then udpates the gradient
            * Stochastic Gradient Descent (SGD)
                * updates after every input sample
                * results in noisy or fluctuating loss outputs
                * can be slow
            * Mini Batch Gradient Descent
                * take a batch of input samples 
                * update gradient after the batch is processed
                * faster convergence to global minima (faster training)
    * Type of Regularization
        * L2 and L1 Regularization
        * Corss validation
        * Early Stopping
        * Drop out
        * Dataset augmentation
    * Epochs Iterations Batch Size
        * Epoch
            * An Epoch occurs when the full set of our training data is passed/forward propagated and then backpropagated through our neural network. through our neural network. 
        * Batches
            * Unless with RAM, can't simply pass all training data to NN in training.
            * Split dataset into segments (batches)
            * Batch size is number of training samples used in each batch
            * Example:
                * 1000 sample data
                * Batch size of 100
                * In training, take 100 samples and use it forward/backward pass, update model weights
                * if batch size is 1. → Stochastic Gradient Descent
        * Iterations
            * Iterations are the number of batches needed to complete one Epoch (full train dataset)
            * Example: 1000 training samples, batch size = 100 → Need 10 iterations to complete one epoch
* Model Evaluation: Loss and Accuracy
    * Accuracy:
    * Loss:
* AB Testing
    * Steps:
        1. Define Hypothesis
        2. How to evaluate experient?
        3. How much of the effect is desired?
        4. How many test subjects needed?
    * Hypothesis
        * H0: There is no difference between A and B (Difference is only from randomness)
        * Ha: B is better than A (Difference has a causation)
    * Evaluation Metric

### Bagging


### Boosting

### Clustering
    * K-means clustering
        * How?
            * choosing k
            * Assign all the points to the closest cluster centroid
            * compute the centroid of the newly formed clusters. 
            * Repeat last two steps until newly formd clusters stoped changing
        * choosing K: Elbow method and silhouette analysis
            * Elbow curve
            * Solhouette Analysis
                * validating the consistency within clusters of data
                *  define the mean distance of a point to all other points in its cluster (a(i)) and also define the mean distance to all other points in the closest cluster (b(i)). 
                * ![equation](https://latex.codecogs.com/gif.latex?s(i)&space;=&space;\frac{b(i)-a(i)}{max(b(i),&space;a(i))})
                    * if near +1: far away from other clusters
                    * if near 0: very close and intersecting other clusters

        * Pros
            * Relatively simple to implement.
            * Scales to large data sets.
            * Guarantees convergence.
            * Can warm-start the positions of centroids.
            * Easily adapts to new examples.
            * Generalizes to clusters of different shapes and sizes, such as elliptical clusters. 

        * Cons
            * We still need to choose K manually
            * It is dependent on initial values
            * Can run into problems when clustering varying sizes and density
            * Sensitive to outliers
            * Doesn’t scale well with large number of dimensions (can be mitigated by using PCA)
            * Only works for numeric values, as such categorical values will either have to be translated into some numerical meaning (e.g. high, medium, low can be mapped to 3,2,1 this can’t always work for categories like types of fruit. Alternatively, we can use K-Medians, or K-Modes to alleviate this issue.

    * Agglomerative Hierarchical clustering
        * How?
            * Bottom up approach
            * treat each data point as a single cluster
            * successively merge or agglomerate pairs of clusters until all clusters have been merged into a single cluster that contains all data points
            * hierarchy of clusters is represented as a tree or referred as a dendrogram
            * root vs. leaf
        * Pros
            * We don’t need to specify the number of clusters, k
            * The algorithm is not too sensitive to the choice of distance metric with all of them working equally well in most cases
            * It works well in uncovering naturally hierarchical data 
        * Cons
            * It is quite slow and doesn’t scale well 
    * Mean shift clustering
        * Sliding window based algorithm
        * Locating centroid of each cluster
        * updating candidates for each center points to be the mean of the points within the sliding window
        * candidate windows are filtered in a post-processing stage to eliminate near duplicates, forming a final set of center points 
    * DBSCAN
        * Similiar to Mean Shift
        * Density based cluster algorithm
        * Pros
            * No preset K number of clusters needs to be set
            * It can identify outliers
            * It can find erratically sized and shaped clusters well 
        * Cons
            * It doesn’t perform well when clusters are of varying densities (due to the setting of ε and minPoints for identifying the neighborhood points, as these will vary from cluster to cluster when they have different densities.
            * High-Dimensional data poses problems when choosing ε

    * Expectation-Maximization Using Gaussian Mixture Model (GMM)
        * Why?
            main weaknesses of K-Means: naïve use of the mean value for the cluster center
        * Gaussian Mixture Models (GMM) allows more flexibility than K-Means by assuming the points are Gaussian Distributed. 
        * have two parameters to describe the cluster shape, the mean and standard deviation
        * Pros
            * GMMs are a lot more flexible in terms of cluster covariance than K-Means
            * the clusters can take on any ellipse shape, rather than being restricted to circles
            * Due to the use of probabilities data points can have multiple clusters e.g. a point can have 0.65 probability in being in one cluster and 0.35 in another. 
        * Cons
            * Choosing K and scaling to higher dimensions 
### PAC: Principle Component Analysis
* What is PCA?
    * compress dataset dimensions from higher to lower
    * based on eigenvectors of the variance in dataset
    * data compression saving loads in processing time, helpful for cluster analysis
    * PCA find a new set of dimension 
        * all dimensions are orthogonal: Linearly independent
        * Ranked according to the variance of data along them. This means the first principal component contains the most information about the data.
    * Cons:
        * PCA works only if the observed variables or data linearly correlate
        * Info will be lost when reducing dimensionality
        * sensitive to scale, all data should be normalized
        * Visualization hard to interpret
### t-SNE t-dsitributed stochastic neighbor embedding
* what is t-SNE
    * another dimensionality reduction algorithm, more effective than PCA
    * Why?
        * PCA unable to capture non-linear dependencies
        * PCA has crowding problem caused by dimenionality reduce. 
        * t-SNE solves this by making optimiztion spread out
    * t-SNE uses stochastic neighbours
    * no clear line between which points are neighbors of other points → lack of boundaries
    * local structure is more important than global structure. allows for a well-balanced dimensionality reduction

## Recommendation Engines
* Why recomendation system important?
    * information overload
    * make selection easier
### Selection Overview
* How do we review items
    * Approach 1: Net score
        * net score = positive rating - negative rating
    * Approach 2: Average Rating
        * Average rating = positive rating / Total rating
    * Approach 3: Correct Score = Lower bound of Wilson score confidence interval for a Bernoulli parameter
* Types of Recommendation systems
    * Collaborative Filtering 
        * recommend products based on the history of user behaviors
        * looks at the similarities between users
        * user to item matrix to find similiar users
            * Cosine Similarity
            ![equation](https://latex.codecogs.com/gif.latex?Cosine&space;Similarity(\boldsymbol{A,B})&space;=&space;cos(\boldsymbol{A,B})=\frac{\boldsymbol{A}\cdot&space;\boldsymbol{B}}{\left&space;\|&space;\boldsymbol{A}&space;\right&space;\|\left&space;\|&space;\boldsymbol{B}&space;\right&space;\|})
            * A and B represent users
            * Ai, Bi, represent each item user A and B purchased
        * Cons:
            * Systems performed poorly when they had many items but comparatively few ratings
            * Computing similarities between all pairs of users was expensive
            * User profiles changed quickly and the entire system model had to be recomputed
            * Sparsity – the number of purchases made by each user is extremely small compared to the number of items that exist.
            * Cold-start issues – we don’t have anything to recommend to new users as they haven’t made any purchases/reviews/likes yet
            * How to recommend a new item?
            * Scalability – the more users grow the larger the computational requirements to calculate distances between them. 
    * Content-based filtering (item-based)
        * calculate similarity between items
        * for systems that have more users than items
* Latent-Factor methods
    To solve scalability issues
    * What and How
        * create a new and most times a reduced feature space of the original user or item vectors, leading to reduced noise and faster computations in real time
    * Two methods:
        * Matrix factorization
            * Popular used in Netflix Prize Competition
            * Singular Value Decomposition (SVD)
                * ![equation](https://latex.codecogs.com/gif.latex?P_{m\times&space;n}&space;=&space;U_{m\times&space;m}\Sigma_{m\times&space;n}V_{n\times&space;n})
                * U and V are unitary matrixes
            * Simon Funk's SVD
                * Issues with SVD
                    * the way missing values imputed can have an undesirable impact on the item
                    * Computational complexity for taining can be high when all entries are considered
                * Solution:
                    only non-missing entries are considered
        * Deep learning

### NLP
NLP in businesses
    * Translating
    * Summarizing information
    * Sentiment Analysis
    * Chatbots
    * Information extraction and search
    * Auto responder and auto complete
#### Bag of Words Model
* Tokenization
    * Tokenization is the process where we take an input string of text and split it up (parse) into individual string elements
    * It is a vectorization technique that allows to represent words as real numbers
* Normalization
    * Case Normalization
    * Stop Word Removal
    common words that do not act additional information or predictive value due to how common they are in normal text. 
    * Removing Punctuation and special symbos
    * Lemmatizing/Stemming
        * reduce inflection forms to normalize words with the same lemma.
        * Lemmatising does this by considering the context of the word while stemming does not. 
        * most current lemmatizers or stemmer libraries are not highly accurate. 
* TF_IDF Vectorizer(Term Frequency - Inverse Document Frequency)
    * account the importance of each item in a document/text
    * TF(i,j) = # of times term i is in document j /  total # of terms in document
    * IDF(i) = ln(total # of documents / # of documents containing term i)
    * TF-IDF(i,j) = TF x IDF
* Word2Vec: efficient estimation of word representation in vector space
    * Word2vec has been pre-trained over large corpora
    * It uses an unsupervised learning method to determine semantic and syntactic meaning from word cooccurrence, which is then used to construct vector representations for every word in the vocabulary. 
    * Two model architectures are used to train word2vec:
    separate the data into observations of target words and their context
        * continuous bag of words
        * skip gram
    * Both use a context window to determine the contextually similar words
    * assumption: words which are used and occur in the same contexts tend to purport similar meaning
    * The context window of Word2Vec is dynamic which has a max size
    * The context is samples from the max window size with a probability of 1/d where d is the distance between the word to target. 
