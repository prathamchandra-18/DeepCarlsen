# Implementation of DeepChess: A Deep Learning Approach to Chess

I came across the research paper on DeepChess and decided to undertake its implementation to gain hands-on experience with TensorFlow and explore the capabilities of Deep Learning in chess.

**How to Play:**
To start playing, first, install python-chess. Then, from the main directory, execute: `python game.py`

**Model Training:**
For training the model, the following setup was used::
- CUDA 7.5
- Tensorflow 0.10.0

To train the model using the available data in the 'pGames' folder, run: python train.py. You can find some older network checkpoints in the 'net' folder.

**Custom Dataset Mining:**
If you wish to mine a different dataset, run Python `get_data.py`, ensuring that you modify the file name in the source code accordingly.
**Key Insights:**
The fundamental concept behind the research paper is to utilize a deep network for playing chess by training it with an evaluation function that takes two positions as input and produces a superior one as output. This way, instead of comparing numerical evaluations of two positions, the network enables a modified Alpha-Beta pruning algorithm to compare the positions themselves.
##
The network consists of two main components: "Pos2Vec" and a fully connected MLP. The "Pos2Vec" component is a Deep Belief Network comprising four stacked autoencoders, trained layer by layer in an unsupervised manner. Two identical "Pos2Vec" components run in parallel and feed into a 4-layer MLP. The entire network was trained on one million pairs of positions, and the pre-training serves as the initial weights for the "Pos2Vec" components.
**Training Dataset:**
The network was trained on the CCRL dataset.(http://www.computerchess.org.uk/ccrl/4040/games.html). 
