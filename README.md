# youtube-trending-predictions

## Installation
To follow the steps used in this project, begin by cloning this repository in a third-party application or by executing the following:

git init

git clone https://github.com/lanewhitmore/youtube-trending-predictions.git

## Contributors
Authors: Caleb McCurdy, Joel Day, Lane Whitmore

Company Name: ViewBoost

Company Industry: Social Media Content & Brand Building

## Methods
Data Exploration

Data Pre-Processing

Data Visualization

Data Modeling (training, testing, validation, metrics)

## Technologies
Microsoft Excel

Google Docs

Google Slides

Microsoft PowerPoint

AWS Services (Athena, Sagemaker, S3)

## Abstract
ViewBoost aims to help high performing YouTube clients increase their view counts and revenue by providing data-driven recommendations for optimizing their videos. These recommendations leverage both Machine learning and trending YouTube information from various regions of the world. 

## Problem Statement
The problem of low engagement and revenue on YouTube affects millions of creators worldwide, and can be attributed to a variety of factors, including low-quality content, ineffective promotion strategies, and a lack of data-driven decision making. This problem is particularly acute for high-performing YouTube clients who are already producing quality content but are not achieving the desired levels of success on the platform.

For ViewBoost, this problem is worth addressing because of the potential benefits that can be derived from a successful YouTube presence, and the subsequent increase in revenue that follows. As an example, increased YouTube views and likes will lead to greater sponsorship opportunities, which leads to more revenue for ViewBoost as the company brings the sponsorships to the client for a percentage of the profit. By helping high-performing YouTube clients optimize their videos and achieve greater success on the platform, ViewBoost has the potential to create significant value for both creators and businesses, and help them achieve their goals in the competitive online video market.

## Goals
1. Be able to identify characteristics of a trending video on YouTube.

2. Predict how a creator’s video will perform on the platform by either predicting likes or views.

3. Help clients make more informed decisions about what types of videos to create.  - Offer insight into other potential profitable avenues, or insight into how profitable a new YouTube sector can be. 

4. Offer long-term growth potential by finding rising topics as well as determining risk levels for different YouTube video features - some features may be more profitable but riskier. 

5. Help clients cultivate a better understanding of their listeners, as well as listeners of similar artists, to increase viewer retention rates and an overall deeper understanding of their viewer base. 


## Non-Goals
1. This project will not be predicting dislikes on videos.

2. This project will be limited to English speaking countries (U.S., Canada & Great Britain) as that is our U.S. based content creators’ primary audience.

3. This project will not be designed to accurately predict views and likes on non-popular videos as the training data only comes from trending videos.

4. This project will not be analyzing the visuals or audio within a video, but rather focuses on statistics describing the content and its results.


## Data Sources
The data comes in the form of a CSV and a JSON file for each region that has been hosted on Kaggle’s website. This dataset is updated daily by the hosting user Rishav Sharma. The CSV for Great Britain, Canada and the U.S. are massive with 34,534, 37,227 and 35,596 records respectively. The JSON files are much smaller to store the video category information, for each region these files are 31 (GB & Canada) and 32 (U.S.) item records long. As the CSVs are large, hosting them will be the first priority in an S3 bucket with the three JSON files. Currently, the intention is to collect this data using a Lambda pipeline and schedule the S3 bucket to update daily or weekly using EventBridge within AWS. 

The bucket will be pathed here to access the YouTube statistics data set: s3://ads508youtubebucket/youtubeStatistics/.

## References
Five Cognitive Biases In Data Science (And how to avoid them). (n.d.). KDnuggets. 
https://www.kdnuggets.com/2020/06/five-cognitive-biases-data-science.html

Sharma, R. (2023). YouTube Trending Video Dataset (updated daily). Kaggle. Retrieved from: 
https://www.kaggle.com/datasets/rsrishav/youtube-trending-video-dataset?select=US_category_id.json

The Risks of AutoML and How to Avoid Them. (2019, October 24). Harvard Business Review. 
https://hbr.org/2019/10/the-risks-of-automl-and-how-to-avoid-them

## Project and Presentation
[Final Project Report.pdf](https://github.com/lanewhitmore/youtube-trending-predictions/files/11258231/Final.Project.Report.pdf)
[Executive Summary and Recommendations.pdf](https://github.com/lanewhitmore/youtube-trending-predictions/files/11258243/Executive.Summary.and.Recommendations.pdf)
