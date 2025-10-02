# Kilter Board Boulder Difficulty Grading

This project explores the use of machine learning to predict the difficulty grade of bouldering routes set on a **Kilter Board**. Using climb data, the models attempt to learn patterns in hold selection and arrangement that correlate with route difficulty, with the ultimate goal of automatically suggesting or validating grades for new problems.


## Project Overview

* **Goal**: Train models that can classify the difficulty of boulder problems based on their configuration on the Kilter Board.
* **Data**: Kilter Board route data, including hold positions and assigned grades.
* **Approach**: Convolutional Neural Network (CNN) architectures and multilayer perceptrons are tested.

The most promising results so far have come from **Convolutional Neural Networks**, which can capture the spatial structure of the board and the relationships between holds.


## Key Files

* `CNN.ipynb`
  This is the primary notebook to reference (see [HERE](src/CNN.ipynb)). It contains the end-to-end pipeline: data preprocessing, CNN architecture definition, training, evaluation, and visualizations of performance.
