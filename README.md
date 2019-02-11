## Final Year Project
- **Description**:
In this project we study how Deep learning can be applied to Robotics. More specifically we use a tactile sensor, the [TacTip](https://softroboticstoolkit.com/tactip) 
to extract images that represent the physical deformation of the tactile sensor whilst in contact with an object. Using these images we are able to
accurately predict the angle and distance of the TacTip with respect to the edge of the object which allows the robot arm to autonomously
follow the edge.
- **Current work**:
We are currently trying to visualise what the network is learning, based on this [paper](https://arxiv.org/abs/1311.2901). This should give
us some intuition into what part of the image is useful to the network and if we could possibly crop the image to reduce the training time
of the network without penalizing the accuracy.
