import os
import numpy as np
import gym
import json
import logging
import random

# Mock implementations of stable-baselines3 classes
class PPO:
    """Mock implementation of PPO from stable-baselines3"""
    def __init__(self, policy, env, learning_rate=0.001, verbose=0, tensorboard_log=None):
        self.policy = policy
        self.env = env
        self.learning_rate = learning_rate
        self.verbose = verbose
        self.tensorboard_log = tensorboard_log
        self.ep_info_buffer = []
        
    def learn(self, total_timesteps, callback=None):
        """Mock implementation of learn method"""
        if callback:
            callback.init_callback(self)
        
        for step in range(total_timesteps):
            # Simulate training progress
            if callback and step % 100 == 0:
                callback.on_step()
                
            # Add episode info every certain number of steps
            if step % 500 == 0:
                self.ep_info_buffer.append({
                    "r": random.uniform(0.2, 0.8),
                    "quest_complexity": random.uniform(0.4, 0.7),
                    "quest_coherence": random.uniform(0.5, 0.9),
                    "quest_novelty": random.uniform(0.3, 0.8),
                    "quest_feasibility": random.uniform(0.6, 0.9)
                })
                if callback:
                    callback.on_rollout_end()
                    
        return self
    
    def predict(self, observation, deterministic=True):
        """Mock implementation of predict method"""
        # Generate a simple action vector
        action = np.array([
            random.randint(0, 4),  # quest type
            random.randint(0, 4),  # num objectives
            random.randint(0, 4),  # difficulty
            random.randint(0, 4),  # reward magnitude
            random.randint(0, 4),  # location preference
            random.randint(0, 4)   # npc involvement
        ])
        return action, None
    
    def save(self, path):
        """Mock implementation of save method"""
        # Create an empty file to simulate saving
        with open(f"{path}.zip", "w") as f:
            json.dump({"mock": "model"}, f)
        return path
    
    @classmethod
    def load(cls, path, env=None):
        """Mock implementation of load method"""
        return cls("MlpPolicy", env)

class BaseCallback:
    """Mock implementation of BaseCallback"""
    def __init__(self, verbose=0):
        self.verbose = verbose
        self.model = None
        self.training_env = None
        self.n_calls = 0
        
    def init_callback(self, model):
        """Initialize callback with model and training env"""
        self.model = model
        self.training_env = model.env
        
    def on_step(self):
        """Called after each step of the environment"""
        self.n_calls += 1
        self._on_step()
        return True
        
    def _on_step(self):
        """Implemented by child classes"""
        pass
        
    def on_rollout_end(self):
        """Called at the end of a rollout"""
        self._on_rollout_end()
        return True
        
    def _on_rollout_end(self):
        """Implemented by child classes"""
        pass

logger = logging.getLogger(__name__)

class QuestGenerationAgent:
    """
    Reinforcement learning agent for quest generation.
    
    This agent uses PPO (Proximal Policy Optimization) algorithm from
    stable-baselines3 to learn effective quest generation policies.
    """
    
    def __init__(self, env, quest_model, model_path="models/quest_gen_model"):
        """
        Initialize the quest generation agent.
        
        Args:
            env: OpenAI Gym compatible environment
            quest_model: Model for quest structure representation
            model_path: Path to save/load trained models
        """
        self.env = env
        self.quest_model = quest_model
        self.model_path = model_path
        self.model = None
        self.training_stats = {
            'rewards': [],
            'quest_complexity': [],
            'quest_coherence': [],
            'quest_novelty': [],
            'quest_feasibility': []
        }
        
        # Create model directory if it doesn't exist
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Try to load a pre-trained model if it exists
        try:
            self.load_model()
            logger.info("Loaded pre-trained model")
        except Exception as e:
            logger.info(f"No pre-trained model found, will create new model: {e}")
            self._initialize_model()
            
    def _initialize_model(self, learning_rate=0.001):
        """Initialize a new RL model."""
        self.model = PPO(
            "MlpPolicy",
            self.env,
            learning_rate=learning_rate,
            verbose=1,
            tensorboard_log="./tensorboard_logs/"
        )
        logger.info("Initialized new PPO model for quest generation")
            
    def train(self, num_episodes=100, learning_rate=0.001, exploration_rate=0.1):
        """
        Train the agent to generate better quests.
        
        Args:
            num_episodes: Number of training episodes
            learning_rate: Learning rate for the optimizer
            exploration_rate: Exploration rate for action selection
            
        Returns:
            Dictionary with training statistics
        """
        logger.info(f"Training quest generation agent for {num_episodes} episodes")
        
        # Reset training stats
        self.training_stats = {
            'rewards': [],
            'quest_complexity': [],
            'quest_coherence': [],
            'quest_novelty': [],
            'quest_feasibility': []
        }
        
        # Create a custom callback to track training progress
        class TrainingCallback(BaseCallback):
            def __init__(self, agent, verbose=0):
                super(TrainingCallback, self).__init__(verbose)
                self.agent = agent
                
            def _on_step(self):
                if self.n_calls % 100 == 0:
                    logger.debug(f"Training step {self.n_calls}")
                return True
            
            def _on_rollout_end(self):
                # Extract information from the env's last info dict
                if hasattr(self.model, "ep_info_buffer") and len(self.model.ep_info_buffer) > 0:
                    ep_reward = np.mean([ep_info["r"] for ep_info in self.model.ep_info_buffer])
                    self.agent.training_stats['rewards'].append(float(ep_reward))
                    
                    # Extract quest metrics if available in the last info dict
                    last_infos = self.model.ep_info_buffer[-1]
                    if "quest_complexity" in last_infos:
                        self.agent.training_stats['quest_complexity'].append(last_infos["quest_complexity"])
                    if "quest_coherence" in last_infos:
                        self.agent.training_stats['quest_coherence'].append(last_infos["quest_coherence"])
                    if "quest_novelty" in last_infos:
                        self.agent.training_stats['quest_novelty'].append(last_infos["quest_novelty"])
                    if "quest_feasibility" in last_infos:
                        self.agent.training_stats['quest_feasibility'].append(last_infos["quest_feasibility"])
                return True
        
        # Initialize or update model with new parameters
        if self.model is None:
            self._initialize_model(learning_rate=learning_rate)
        else:
            self.model.learning_rate = learning_rate
        
        # Train the model
        callback = TrainingCallback(self)
        self.model.learn(
            total_timesteps=num_episodes * self.env.max_steps,
            callback=callback
        )
        
        # Save the trained model
        self.save_model()
        
        # Save training stats
        self._save_training_stats()
        
        return self.training_stats
    
    def generate_quest(self, complexity=0.5, theme="fantasy", deterministic=True):
        """
        Generate a quest using the trained RL policy.
        
        Args:
            complexity: Desired quest complexity (0.0-1.0)
            theme: Quest theme (fantasy, sci-fi, etc.)
            deterministic: Whether to use deterministic policy
            
        Returns:
            Generated quest as a dictionary
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first or load a pre-trained model.")
        
        # Reset the environment to ensure fresh state
        obs = self.env.reset()
        
        # Modify initial state to include complexity preference
        # We assume the first element of the state is the current quest complexity
        obs[0] = complexity
        
        # Follow the policy to generate a quest
        done = False
        quest_actions = []
        quest_states = [obs.tolist()]
        total_reward = 0
        
        while not done:
            # Use the model to predict the next action
            action, _ = self.model.predict(obs, deterministic=deterministic)
            quest_actions.append(action.tolist())
            
            # Take the action in the environment
            obs, reward, done, info = self.env.step(action)
            quest_states.append(obs.tolist())
            total_reward += reward
        
        # Generate quest structure from actions using the quest model
        quest = self.quest_model.create_quest_from_actions(
            quest_actions, 
            complexity=complexity,
            theme=theme
        )
        
        # Add evaluation metrics
        quest['metrics'] = {
            'complexity': float(self.env.quest_complexity),
            'coherence': float(self.env.quest_coherence),
            'novelty': float(self.env.quest_novelty),
            'feasibility': float(self.env.quest_feasibility),
            'total_reward': float(total_reward)
        }
        
        logger.info(f"Generated quest with complexity {quest['metrics']['complexity']:.2f} "
                   f"and total reward {total_reward:.2f}")
        
        return quest
    
    def evaluate_quest(self, quest_data):
        """
        Evaluate a quest based on learned policy.
        
        Args:
            quest_data: Quest data structure
            
        Returns:
            Evaluation metrics as a dictionary
        """
        # Convert quest to actions
        actions = self.quest_model.convert_quest_to_actions(quest_data)
        
        # Reset environment
        obs = self.env.reset()
        
        # Apply actions and track rewards
        total_reward = 0
        metrics = {}
        
        for action in actions:
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            # Save the final state's metrics
            if done:
                metrics = {
                    'complexity': float(info['quest_complexity']),
                    'coherence': float(info['quest_coherence']),
                    'novelty': float(info['quest_novelty']),
                    'feasibility': float(info['quest_feasibility']),
                    'total_reward': float(total_reward)
                }
        
        return metrics
    
    def save_model(self, path=None):
        """Save the trained model."""
        if self.model is None:
            logger.warning("No model to save")
            return
            
        save_path = path or self.model_path
        self.model.save(save_path)
        logger.info(f"Model saved to {save_path}")
    
    def load_model(self, path=None):
        """Load a trained model."""
        load_path = path or self.model_path
        if os.path.exists(load_path + ".zip"):
            self.model = PPO.load(load_path, env=self.env)
            logger.info(f"Model loaded from {load_path}")
        else:
            raise FileNotFoundError(f"No model found at {load_path}")
    
    def _save_training_stats(self):
        """Save training statistics to file."""
        stats_path = os.path.join(os.path.dirname(self.model_path), "training_stats.json")
        with open(stats_path, 'w') as f:
            json.dump(self.training_stats, f)
        logger.info(f"Training stats saved to {stats_path}")
