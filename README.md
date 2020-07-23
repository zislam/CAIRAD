# CAIRAD
Implements the CAIRAD techique for detecting noisy values in a dataset. Does this with an analysis of coappearance between values. Can output whether or not a record is noisy (i.e. includes 1 or more noisy values), or remove all noisy values and replace them with missing values. 

CAIRAD specification from:

*Rahman, M. G., Islam, M. Z., Bossomaier, T., & Gao, J. (2012, June). Cairad: A co-appearance based analysis for incorrect records and attribute-values detection.* In The 2012 International Joint Conference on Neural Networks (IJCNN) (pp. 1-10). IEEE. Available at [http://doi.org/10.1109/ijcnn.2012.6252669](http://doi.org/10.1109/ijcnn.2012.6252669)

For more information, please see Associate Professor Zahid Islam's website [here](http://csusap.csu.edu.au/~zislam/)

## Installation
Either download CAIRAD from the Weka package manager, or download the latest release from the "**Releases**" section on the sidebar of Github. A video showing the installation and use of the package can be found [here](https://www.youtube.com/watch?v=JUyKobCxruI)

## Compilation / Development
This repository houses a Netbeans project. Load the project into Netbeans to work on the package. Alternatively, download CAIRAD.java and import it into your Weka project to use it in your code.

## Valid options are:

`-T`
coappearanceThreshold - Coappearance Threshold, tau in original paper.

`-L`
coappearanceScoreThreshold - Coappearance Score Threshold, lambda in original paper.

`-M`
makeNoisyMissing - Make detected noise into missing values. 
