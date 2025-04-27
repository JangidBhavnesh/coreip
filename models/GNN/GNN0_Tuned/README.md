GNN0 with tuned hyperparameters (num layers, nodes, aggregation function, learning rate)

GNN0_Best_Tuned_Hyperparameters.pt is the saved model
BEST_Model_Error.csv gives the validation and test MAE for the best iteration of the model
GNN0_Training_Results.csv gives the train/validation/test MAE as a function of epoch
TEST_Predictions.csv gives the predicted values for the test set in the form [index, predicted, actual, error, absolute error], sorted by absolute error. 

Highest error system is a element/orbital combination which is non-existant in training data. The next three are the system which has two very different values in the literature, so the reported average may be of poor quality.

This training was done with the intial graph_data.json data.
