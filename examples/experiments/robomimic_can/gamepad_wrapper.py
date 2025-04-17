import gymnasium as gym
import numpy as np
import pygame

class GamepadExpert:
    """
    A gamepad interface that:
     - polls a joystick via pygame
     - maintains a 'gear' system for translation and rotation speeds
    """
    def __init__(self, 
                 translation_speeds=None,
                 rotation_speeds=None,
                 dead_zone=0.15):
        pygame.init()
        pygame.joystick.init()
        self.joystick = pygame.joystick.Joystick(0)
        self.joystick.init()

        # Default gears if none provided
        # Feel free to change these values or their lengths
        if translation_speeds is None:
            translation_speeds = [0.05, 0.1, 0.2, 0.5, 0.75, 1.0]
        if rotation_speeds is None:
            rotation_speeds = [0.05, 0.1, 0.2, 0.5, 0.75, 1.0]

        self.translation_speeds = translation_speeds
        self.rotation_speeds    = rotation_speeds

        # Start on fastest
        self.trans_idx = len(self.translation_speeds)-1
        self.rot_idx   = len(self.rotation_speeds)-1

        self.dead_zone = dead_zone

    def apply_dead_zone(self, value, DEAD_ZONE=0.15):
        if abs(value) < DEAD_ZONE:
            return 0.0
        elif value > 0:
            return (value - DEAD_ZONE) / (1 - DEAD_ZONE)  # Normalize the positive range
        else:
            return (value + DEAD_ZONE) / (1 - DEAD_ZONE)  # Normalize the negative range

    def update_speeds_via_dpad(self):
        """
        Checks the D-pad (hat) state each frame and updates
        the translation/rotation speed indices accordingly.
        """
        # Most controllers only have one hat; index 0
        # dpad_x in [-1, 0, 1], dpad_y in [-1, 0, 1]
        dpad_x, dpad_y = self.joystick.get_hat(0)

        # Left/right changes translation gear
        if dpad_x > 0:  # D-pad right
            self.trans_idx = min(self.trans_idx + 1, len(self.translation_speeds) - 1)
        elif dpad_x < 0:  # D-pad left
            self.trans_idx = max(self.trans_idx - 1, 0)

        # Up/down changes rotation gear
        if dpad_y > 0:  # D-pad up
            self.rot_idx = min(self.rot_idx + 1, len(self.rotation_speeds) - 1)
        elif dpad_y < 0:  # D-pad down
            self.rot_idx = max(self.rot_idx - 1, 0)

    def get_translation_speed(self):
        return self.translation_speeds[self.trans_idx]

    def get_rotation_speed(self):
        return self.rotation_speeds[self.rot_idx]

    def get_action(self):
        """
        Poll the gamepad and construct a 6D action for the robot arm:
          [tx, ty, tz, rx, ry, rz]
        Also returns (left_bumper_pressed, right_bumper_pressed).
        """
        # Update internal states for the D-pad gear system
        self.update_speeds_via_dpad()

        # Process joystick events so states are updated
        pygame.event.pump()

        # Axes (these indices may vary by controller)
        left_stick_x = self.apply_dead_zone(self.joystick.get_axis(0))   # Left horizontal
        left_stick_y = self.apply_dead_zone(self.joystick.get_axis(1))   # Left vertical
        right_stick_x = self.apply_dead_zone(self.joystick.get_axis(3))  # Right horizontal
        right_stick_y = self.apply_dead_zone(self.joystick.get_axis(4))  # Right vertical
        trigger_l     = self.apply_dead_zone(self.joystick.get_axis(2))  # Left trigger
        trigger_r     = self.apply_dead_zone(self.joystick.get_axis(5))  # Right trigger

        max_speed = self.get_translation_speed()
        rot_speed = self.get_rotation_speed()

        # Basic translation controls
        move_forward_backward = -left_stick_y * max_speed
        move_left_right       =  -left_stick_x * max_speed
        move_up_down          = (trigger_r - trigger_l) * max_speed

        # For demonstration: left bumper (LB=4) toggles "yaw" mode
        is_roll_mode = self.joystick.get_button(4)
        if is_roll_mode:
            roll = right_stick_x * rot_speed
            yaw  = 0.0
        else:
            roll = 0.0
            yaw  = -right_stick_x * rot_speed

        pitch = right_stick_y * rot_speed

        # Create the expert action for the 6D arm
        expert_a = np.array([move_forward_backward, 
                             move_left_right, 
                             move_up_down, 
                             roll, 
                             pitch, 
                             yaw], dtype=np.float32)

        # For gripper: left bumper (LB=4) or right bumper (RB=5).
        # You could also use separate buttons, e.g. LB=4, RB=5
        # Gripper control with A button (button 0) for toggling
        left_bumper  = bool(self.joystick.get_button(2))
        right_bumper = bool(self.joystick.get_button(0))

        return expert_a, (left_bumper, right_bumper)


class GamepadIntervention(gym.ActionWrapper):
    """
    A wrapper that uses a gamepad "expert" to override policy actions
    if the user provides input beyond a small threshold or presses bumpers.

    Includes a gear system for changing translation/rotation speeds with the D-pad.
    Now supports DoF conversion to handle lower dimensional action spaces.
    """
    def __init__(self, env, action_indices=None, ee_dof=None):
        super().__init__(env)

        # Store DoF configuration if specified
        self.ee_dof = ee_dof
        
        # Set up action mapping based on DoF or detect from action space
        if ee_dof is not None:
            if ee_dof == 3:  # [dx, dy, dz, grasp]
                self._map = [0, 1, 2, 6]
                self.expected_action_dim = 4
            elif ee_dof == 4:  # [dx, dy, dz, yaw, grasp]
                self._map = [0, 1, 2, 5, 6]
                self.expected_action_dim = 5
            elif ee_dof == 6:  # [dx, dy, dz, roll, pitch, yaw, grasp]
                self._map = [0, 1, 2, 3, 4, 5, 6]
                self.expected_action_dim = 7
            else:
                raise ValueError("ee_dof must be 3, 4, or 6")
        else:
            # Try to infer from action space if DoF not explicitly provided
            action_dim = self.action_space.shape[0]
            if action_dim == 4:
                self._map = [0, 1, 2, 6]
                self.expected_action_dim = 4
                self.ee_dof = 3
            elif action_dim == 5:
                self._map = [0, 1, 2, 5, 6]
                self.expected_action_dim = 5
                self.ee_dof = 4
            elif action_dim == 7:
                self._map = [0, 1, 2, 3, 4, 5, 6]
                self.expected_action_dim = 7
                self.ee_dof = 6
            else:
                # Default to full dimensionality if we can't determine
                self._map = list(range(action_dim))
                self.expected_action_dim = action_dim
                print(f"Warning: Could not determine DoF from action space shape {self.action_space.shape}. Using default mapping.")

        # Check if environment action space includes a gripper
        self.gripper_enabled = (self.expected_action_dim > self.ee_dof)

        # Create our gamepad "expert" interface
        self.expert = GamepadExpert()

        # Track left/right bumper presses for the info dict
        self.left, self.right = False, False

        # If you only want to override certain action indices, specify them here
        self.action_indices = action_indices
        
        print(f"GamepadIntervention initialized with ee_dof={self.ee_dof}, expected_action_dim={self.expected_action_dim}")
        print(f"Action mapping: {self._map}")

    def action(self, action: np.ndarray):
        """
        Decide whether to override the incoming 'action' (policy action)
        with the user's gamepad input. Handles DoF conversion as needed.
        """
        expert_a, buttons = self.expert.get_action()
        self.left, self.right = buttons

        # Determine if user is actively intervening
        intervened = False
        # If there's meaningful input on the sticks/triggers, interpret it as user intervention
        if np.linalg.norm(expert_a) > 0.001:
            intervened = True

        # Process gripper action
        if self.gripper_enabled:
            # Use bumpers to close/open gripper
            if self.left:  # close gripper
                gripper_action = np.random.uniform(-1, -0.9, size=(1,))
                intervened = True
            elif self.right:  # open gripper
                gripper_action = np.random.uniform(0.9, 1, size=(1,))
                intervened = True
            else:
                gripper_action = np.zeros((1,))
            
            # For full 7D action:
            full_expert_a = np.zeros(7, dtype=np.float32)
            
            # Fill in the available DoFs
            if self.ee_dof == 3:  # Just xyz, no rotation
                full_expert_a[0:3] = expert_a[0:3]  # xyz translation
            elif self.ee_dof == 4:  # xyz + yaw
                full_expert_a[0:3] = expert_a[0:3]  # xyz translation
                full_expert_a[5] = expert_a[5]      # yaw only
            else:  # 6 DoF - full pose control
                full_expert_a[0:6] = expert_a[0:6]  # All translation and rotation
                
            # Add gripper
            full_expert_a[6] = gripper_action[0]
            
            # Now extract only the DoFs we need for the current environment
            expert_a = full_expert_a[self._map]
        else:
            # For environments without gripper, just take the appropriate DoFs
            full_expert_a = np.zeros(6, dtype=np.float32)
            if self.ee_dof == 3:
                full_expert_a[0:3] = expert_a[0:3]
            elif self.ee_dof == 4:
                full_expert_a[0:3] = expert_a[0:3]
                full_expert_a[5] = expert_a[5]
            else:
                full_expert_a = expert_a
                
            expert_a = full_expert_a[:self.expected_action_dim]

        # If only certain indices should be overwritten:
        if self.action_indices is not None:
            filtered_expert_a = np.zeros_like(expert_a)
            filtered_expert_a[self.action_indices] = expert_a[self.action_indices]
            expert_a = filtered_expert_a

        # Check if action dimensions match what we expect
        # This can help debug environment wrapper issues
        if action.shape != (self.expected_action_dim,):
            print(f"Warning: Expected action shape {(self.expected_action_dim,)}, got {action.shape}")
            # Try to adapt the action if possible
            if len(action) == 7 and self.expected_action_dim < 7:
                # We got a full 7D action but expect less - extract what we need
                action = action[self._map]
            elif len(action) < 7 and self.expected_action_dim == 7:
                # We need a full 7D action but got less - expand with zeros
                full_action = np.zeros(7, dtype=np.float32)
                full_action[self._map[:len(action)]] = action
                action = full_action

        # If user intervened, return the user (expert) action, else the policy's
        if intervened:
            return expert_a, True
        else:
            return action, False

    def step(self, action):
        """
        Called each environment step. We produce the possibly overridden action.
        """
        new_action, replaced = self.action(action)
        obs, rew, done, truncated, info = self.env.step(new_action)

        # If we replaced the action, record that in info
        if replaced:
            info["intervene_action"] = new_action

        # Store left/right bumper usage
        info["left"] = self.left
        info["right"] = self.right

        # Store current speeds in the info dict (useful for logging)
        info["translation_speed"] = self.expert.get_translation_speed()
        info["rotation_speed"]    = self.expert.get_rotation_speed()

        return obs, rew, done, truncated, info