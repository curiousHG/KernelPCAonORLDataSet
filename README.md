## Kernel PCA on ORL dataset
In this project we look at how KernelPCA compares to PCA with using different parameters like number of components width of gaussian kernel and degree of polynomial kernel.and
The project has two components a feature extraction step and classification step.
This project is majorly about feature extraction using PCA and KernelPCA.
For classification step we use the kNearest Neighbour classifier
We have used ORLDataset in which a total 40 subjects are there with each having 10 images.
The project consists of a total of 4-experiments:-
    Three - Experiments:
        Taking 1 image of each subject as training sample and doing feature extraction in following ways:-
        1) Using Gaussian kernel-pca, Varying value of kernel width(best = 3000) and number of components(best=30, 10  or 20 are fine).
        2) Using polynomial kernel-pca, varying the degree(better lower example best 2) of polynomial and number of features(20 or 30 are good enough
        3) using PCA , number of principal components (best 30)

    Final experiment:
        Comparing both pca and kernel-pca, varying number of images per person used for training.

To run the file:-
1. Open the code folder in terminal.
2. Check if python is installed using "python -V"
3. Run the command "pip install -r requirements.txt"
4. Then go to experiments folder and run the command "python experiment1.py"
5. Repeat step 4 for each experiment by changing experiment name(experiment2,3 and finalExperiment.py) and all the results will be obtained.