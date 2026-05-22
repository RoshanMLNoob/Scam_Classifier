# Scam_Classifier
This small Project was build by me during learning Machine Learning as of 3:12 AM IST 23 May 2026 this project is finished.
The Main file here is the "Scam_Finder.py" and if one dosnt want to do additional training on it then only that file
with the thetas.csv are essential to for the program to run.

How It Works??
It uses a Multidimentional Linear Classifier which is trained on a classical Perceptron algorithm, the data.csv file contains commonly used words in spam emails/messages so there is a list of that. 
The model takes the user input and then seprates it word by word in lowercase and then passes it through the linear classifier which classifies it based on
the parameters which it takes from the last line of thetas.csv as th and th0, both of which govern the results.

How to use it effectivly-
If you just want to see how it classifies then click run and enter your data to be classified.
But if you want to use the "check_diff_T(T=x)" function with x as your parameter then First you have to open the thetas.csv file manually and just at the last line write x, or under the T column write the T you specified and then a comma this would prevent any erroes.
