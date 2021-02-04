<p align="center">
  <img src="https://github.com/DavideStenner/Kaggle/blob/master/Santa%202020%20-%20The%20Candy%20Cane%20Contest/santa_contest_banner.PNG" />
</p>

This challenge is a two player game and consists of taking the more candy as possible from 100 different canes. Each Cane has a probability of giving a candy and each time it is
selected by one player its probability decrease.

Our approach constists of using a Lightgbm to estimate the expected probability of success for each cane.

We scraped top agent replay using the Kaggle API by collecting over 25.000 matches. Due to memory issue we create different notebook (more than 12) to scrape the data and create the dataset.

The different notebooks are:

- agent-scrape-everything-step-*: This notebook scrapes the different match using the Kaggle API.
The dictionary winner_dic saves the list of the match which has been scraped yet and the winner list, which is used during the generation of the 
dataset.

- generate-everything-step-*: This notebook used the scraped match and generate the training dataset by creating the feature for each bandit in every round.
Each match will have num_bandit * number_round = 100 * 1999 = 199900 rows and 18 columns.

- mapping_sub_score: is a notebook which saves the final score for each agent. This score is used as a weight during the Lightgbm train.

- lgbm-train: in this notebook we trained the Lightgbm model using the score of the agent as a weight for each row.

- lgbm-agent: This is the submitted agent, which uses a greedy method based on the Lightgbm evaluation of each bandit expected reward.
