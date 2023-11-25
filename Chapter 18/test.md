obs:
    position(0.0 = center),
    velocity (positive means right), 
    the angle of the pole (0.0 = vertical, positive means tilt down to the right),
    angular velocity (positive means clockwise)

action:
    left: 0
    right: 1
    
How to read transition_probabilities and rewards:
    transition_probabilities[a][b][c] mean the probability if we stay at 
    state a-th, do the b-th action then end up in state c-th.
    Similar to reward.