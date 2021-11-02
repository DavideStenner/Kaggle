<p align="center">
  <img src="https://github.com/DavideStenner/Kaggle/blob/master/Tabular Playground Series - Oct 2021/tabular_contest_banner.PNG" />
</p>

I use three step in this challenge:

 - Optuna with lightgbm, catboost, xgboost, and nn to find best parameter.
 - Pseudo ensemble on every model
 - Ensemble with simple average
 
 
I discritize the input before passing to the nn by creating 256 bin on every numerical feature.