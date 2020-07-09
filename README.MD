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