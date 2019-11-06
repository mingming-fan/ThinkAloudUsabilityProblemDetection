# ThinkAloudUsabilityProblemDetection

## Explanation on evaluation files and how to use them:

eval_tenFoldCross.py performs ten-fold cross valiation on the whole data set, specify a model name in order to run it, a model name has to be from one of **'rf' (random forest)** or **'svm' (support vector machine)** i.e.

`python ./eval_tenFoldCross.py svm`

eval_userOut.py performs leave-one-user-out evaluation on data of a particular product thus a specified product name is required, a product name has to be from **['history', 'science', 'machine', 'remote']** i.e.

`python ./eval_userOut.py svm science`

eval_productOut.py performs leave-one-product-out evaluation on data of a particular user thus a specified user name is required, a user name has to be an integer from **range 1 to 8**, i.e.

`python ./eval_productOut.py svm 2`

Inside CNN and RNN folds, there are two training files to run the evaluation, respectively. Change directory to CNN or RNN and evaluation files can be called directly:

`python ./train_CNN.py` or
`python ./train_rnn.py`
