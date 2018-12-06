import numpy as np
from physics_sim import PhysicsSim

class Task():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, init_velocities=None, 
        init_angle_velocities=None, runtime=5., target_pos=None):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime) 
        self.action_repeat = 3

        self.state_size = self.action_repeat * 6
        self.action_low = 0
        self.action_high = 900
        self.action_size = 4
        
        self.last_pose = self.sim.pose

        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.]) 

    def get_reward(self):
        """Uses current pose of sim to return reward."""
        
        #reward = -30*np.tan(0.01*(abs(self.sim.pose[:3] - self.target_pos).sum())+1.65)-300
        
        reward = -3*np.tanh(0.5*(abs(self.sim.pose[:3] - self.target_pos).sum())-10)
        #Penalize crashing with the ground
        if self.sim.done and self.sim.runtime > self.sim.time and self.sim.pose[2] == 0:
            reward -= 10
            
        #Penalize for any velocity in z axis for any velocity the penalization increse when near the ground z=0
        reward -= 0.0009*((self.sim.v[2]**2)/(self.sim.pose[2]+1))
        
        
        
        
        #reward = -1*np.tanh(0.01*(abs(self.sim.pose[:3] - self.target_pos).sum())-2)
        #reward = -30*np.tan(0.01*(abs(self.sim.pose[:3] - self.target_pos).sum())+1.65)-300
        #reward = 1.-.3*(abs(self.sim.pose[:3] - self.target_pos)).sum()
        
#         # Penalize going far from the objective.       
#         if (abs(self.sim.pose[:3] - self.target_pos)).sum() > (abs(self.last_pose[:3] - self.target_pos)).sum():
#             reward -= .015*(abs(self.sim.pose[:3] - self.target_pos)).sum()
#             self.last_pose = self.sim.pose[:3]
            
#         #Penalize crashing with the ground
#         if self.sim.done and self.sim.runtime > self.sim.time:
#             reward -= 20
        
        return reward

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += self.get_reward() 
            pose_all.append(self.sim.pose)
        next_state = np.concatenate(pose_all)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.sim.pose] * self.action_repeat) 
        return state