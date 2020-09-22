# VideoSynchronizationWithPytorch
Code of the paper 'End-To-End Deep Learning Model for Cardiac Cycle Synchronization from Multi-View Angiographic Sequences' (presented at EMBC 2020) using Pytorch.

## Evaluate a model
1. Open a Jupyter Notebook server with the command `jupyter notebook`.
2. Open the `main.ipynb` notebook within the Jupyter Notebook server.
3. Run the following cells:
	- Model declaration
	- Load model (within Test trained model)
	- Load test set (also within Test trained model)
	- Compute distance and similarity matrices for each video
	- Compute distance and similarity matrices for video comparison
	- Pathfinding
	- Global pathfinding result
The score will be shown next to the `Average total score (Combination)` text.

### Test on new data
1. Follow the instruction in the README of the AngioMatch repo to prepare the data.
2. Open a Jupyter Notebook server with the command `jupyter notebook`.
3. Open the `main.ipynb` notebook within the Jupyter Notebook server.
4. Modify the paths in the "Load test set" cell.
5. Execute the steps needed to evaluate the model.

## Train a model on new data
1. Follow the instruction in the README of the AngioMatch repo to prepare the data.
2. Open a Jupyter Notebook server with the command `jupyter notebook`.
3. Open the `main.ipynb` notebook within the Jupyter Notebook server.
4. Modify the parameters of the model as you wish and the experiment path in the "Model declaration" cell.
5. Modify the validation_paths and test_paths variable of the "Angio sequence soft multi siamese" cell of the "Load dataset" section.
6. Copy paste the test_paths variable to the "Load test set" cell.
6. Run the following cells:
	- Model declaration
	- Angio sequence soft multi siamese (within Load dataset)
	- Angio sequence multisiamese (within Train)
	- All the ones needed for evaluating a model starting from the third one (Load test set)

## Run an hyperparameter search
1. Make sure the right validation and test sets contain the right paths in the `load_training_set()` method of the `hyperparameter_search.py` file.
2. If you made changes to the `main.ipynb` notebook, replicate those changes in the `hyperparameter_search.py` file.
3. Review the content of the `random_parameters` variable at the top of the `hyperparameter_search.py` file.
4. Run the `run_hyperparameter_search.bat` script.
5. At any time, you can run the `read_hyperparameter_search_results.py` file to show the results of the hyperparameter search.
