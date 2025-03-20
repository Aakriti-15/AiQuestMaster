import gym
import numpy as np
from gym import spaces
import logging

logger = logging.getLogger(__name__)

class GameEnvironment(gym.Env):
    """
    OpenAI Gym compatible environment for quest generation.
    
    This environment simulates a game world where quests can be generated.
    It provides the state representation, action space, and reward function
    for the reinforcement learning agent to learn from.
    """
    
    def __init__(self, 
                 state_size=20,  # Size of the state vector representing game world
                 max_steps=50,   # Maximum steps per episode
                 world_complexity=0.6):  # Complexity of the simulated game world
        super(GameEnvironment, self).__init__()
        
        self.state_size = state_size
        self.max_steps = max_steps
        self.world_complexity = world_complexity
        self.current_step = 0
        
        # Define action space for quest generation
        # Actions represent different decisions in quest creation:
        # - Quest type selection (0-4)
        # - Number of objectives (0-4)
        # - Difficulty level (0-4)
        # - Reward magnitude (0-4)
        # - Location selection (0-4)
        # - NPC involvement (0-4)
        # Each action dimension has 5 discrete options (0-4)
        self.action_space = spaces.MultiDiscrete([5, 5, 5, 5, 5, 5])
        
        # Observation space (state space)
        # Represents the current state of the game world and quest progress
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(self.state_size,), dtype=np.float32
        )
        
        # Initialize the state
        self.state = None
        
        # Quest-specific parameters
        self.quest_complexity = 0
        self.quest_coherence = 0
        self.quest_novelty = 0
        self.quest_feasibility = 0
        
        # World state variables
        self.world_state = {
            'locations': [],
            'npcs': [],
            'items': [],
            'quests': []
        }
        
        # Initialize the environment
        self.reset()
        
    def reset(self):
        """Reset the environment to initial state for new episode."""
        self.current_step = 0
        
        # Initialize game world state with some randomness to ensure variety
        self._initialize_world()
        
        # Reset quest quality metrics
        self.quest_complexity = 0
        self.quest_coherence = 0
        self.quest_novelty = 0
        self.quest_feasibility = 0
        
        # Initial state is a combination of world state and quest metrics
        self.state = self._get_state()
        
        return self.state
    
    def step(self, action):
        """
        Take an action to progress the quest generation process.
        
        Args:
            action: Array of discrete actions for different quest aspects
            
        Returns:
            next_state: New state after action
            reward: Reward for this action
            done: Whether episode is complete
            info: Additional information
        """
        self.current_step += 1
        
        # Apply the chosen action to update the quest being generated
        self._apply_action(action)
        
        # Get the new state
        next_state = self._get_state()
        
        # Calculate reward based on quest quality
        reward = self._calculate_reward()
        
        # Check if episode is done
        done = self.current_step >= self.max_steps
        
        # Additional info
        info = {
            'quest_complexity': self.quest_complexity,
            'quest_coherence': self.quest_coherence,
            'quest_novelty': self.quest_novelty,
            'quest_feasibility': self.quest_feasibility,
            'step': self.current_step
        }
        
        self.state = next_state
        return next_state, reward, done, info
    
    def _initialize_world(self):
        """Initialize the game world state with locations, NPCs, items."""
        # Generate random number of locations (3-10)
        num_locations = np.random.randint(3, 11)
        self.world_state['locations'] = [
            {
                'id': i,
                'difficulty': np.random.uniform(0, 1),
                'accessibility': np.random.uniform(0, 1),
                'population': np.random.uniform(0, 1)
            }
            for i in range(num_locations)
        ]
        
        # Generate random number of NPCs (5-15)
        num_npcs = np.random.randint(5, 16)
        self.world_state['npcs'] = [
            {
                'id': i,
                'friendliness': np.random.uniform(0, 1),
                'importance': np.random.uniform(0, 1),
                'location_id': np.random.randint(0, num_locations)
            }
            for i in range(num_npcs)
        ]
        
        # Generate random number of items (5-20)
        num_items = np.random.randint(5, 21)
        self.world_state['items'] = [
            {
                'id': i,
                'value': np.random.uniform(0, 1),
                'rarity': np.random.uniform(0, 1),
                'location_id': np.random.randint(0, num_locations)
            }
            for i in range(num_items)
        ]
        
    def _apply_action(self, action):
        """
        Apply the selected action to update the quest being constructed.
        
        Args:
            action: Array of discrete values representing quest generation decisions
        """
        # Unpack action components
        quest_type = action[0]  # 0: Fetch, 1: Kill, 2: Escort, 3: Discover, 4: Crafting
        num_objectives = action[1] + 1  # 1-5 objectives
        difficulty = action[2] / 4.0  # Normalize to 0-1
        reward_magnitude = action[3] / 4.0  # Normalize to 0-1
        location_preference = action[4]  # Preferred location type
        npc_involvement = action[5]  # NPC involvement level
        
        # Update quest complexity based on number of objectives and difficulty
        self.quest_complexity = (0.6 * num_objectives / 5.0) + (0.4 * difficulty)
        
        # Update quest coherence based on how well components fit together
        # Higher coherence if location matches quest type and appropriate NPC involvement
        location_match = 1.0 - (abs(location_preference - quest_type) / 4.0)
        npc_match = 1.0 - (abs(npc_involvement - quest_type) / 4.0)
        self.quest_coherence = (0.5 * location_match) + (0.5 * npc_match)
        
        # Calculate novelty compared to existing quests
        if len(self.world_state['quests']) > 0:
            similarities = []
            for quest in self.world_state['quests']:
                similarity = (
                    int(quest['type'] == quest_type) +
                    (1.0 - abs(quest['difficulty'] - difficulty)) +
                    (1.0 - abs(quest['reward'] - reward_magnitude)) +
                    int(quest['location_type'] == location_preference) +
                    int(quest['npc_involvement'] == npc_involvement)
                ) / 5.0
                similarities.append(similarity)
            
            avg_similarity = np.mean(similarities)
            self.quest_novelty = 1.0 - avg_similarity
        else:
            # First quest is considered novel
            self.quest_novelty = 1.0
        
        # Calculate feasibility based on difficulty and world state
        world_difficulty_capacity = np.mean([loc['difficulty'] for loc in self.world_state['locations']])
        self.quest_feasibility = 1.0 - abs(difficulty - world_difficulty_capacity)
        
        # Add the current quest to the world's quest list
        current_quest = {
            'type': quest_type,
            'objectives': num_objectives,
            'difficulty': difficulty,
            'reward': reward_magnitude,
            'location_type': location_preference,
            'npc_involvement': npc_involvement
        }
        
        self.world_state['quests'].append(current_quest)
    
    def _get_state(self):
        """Generate the state vector based on current world and quest state."""
        # Create state vector with relevant features
        state = np.zeros(self.state_size, dtype=np.float32)
        
        # First 4 elements: Quest quality metrics
        state[0] = self.quest_complexity
        state[1] = self.quest_coherence
        state[2] = self.quest_novelty
        state[3] = self.quest_feasibility
        
        # Next elements: World state information (simplified)
        num_locations = len(self.world_state['locations'])
        num_npcs = len(self.world_state['npcs'])
        num_items = len(self.world_state['items'])
        num_quests = len(self.world_state['quests'])
        
        # Normalize by expected maximum values
        state[4] = num_locations / 10.0  # Assume max 10 locations
        state[5] = num_npcs / 15.0  # Assume max 15 NPCs
        state[6] = num_items / 20.0  # Assume max 20 items
        state[7] = num_quests / 10.0  # Assume max 10 quests
        
        # World complexity
        state[8] = self.world_complexity
        
        # Progress through episode
        state[9] = self.current_step / self.max_steps
        
        # Remaining elements can be used for quest-specific features
        if num_quests > 0:
            latest_quest = self.world_state['quests'][-1]
            state[10] = latest_quest['type'] / 4.0
            state[11] = (latest_quest['objectives'] - 1) / 4.0
            state[12] = latest_quest['difficulty']
            state[13] = latest_quest['reward']
            state[14] = latest_quest['location_type'] / 4.0
            state[15] = latest_quest['npc_involvement'] / 4.0
        
        # Fill any remaining state elements with random noise to encourage exploration
        for i in range(16, self.state_size):
            state[i] = np.random.uniform(0, 0.1)  # Small noise
            
        return state
    
    def _calculate_reward(self):
        """Calculate reward based on quest quality metrics."""
        # Base reward is determined by a weighted combination of quest metrics
        base_reward = (
            0.3 * self.quest_complexity +  # Reward complexity but don't overweight it
            0.3 * self.quest_coherence +   # Coherent quests are important
            0.2 * self.quest_novelty +     # Novel quests are good but less important than coherence
            0.2 * self.quest_feasibility   # Feasible quests are necessary
        )
        
        # Penalties for extreme values
        penalties = 0
        
        # Penalty for too simple or too complex quests
        if self.quest_complexity < 0.2:
            penalties += 0.5 * (0.2 - self.quest_complexity)
        elif self.quest_complexity > 0.8:
            penalties += 0.3 * (self.quest_complexity - 0.8)
            
        # Severe penalty for incoherent quests
        if self.quest_coherence < 0.4:
            penalties += 0.7 * (0.4 - self.quest_coherence)
            
        # Mild penalty for similar quests
        if self.quest_novelty < 0.3:
            penalties += 0.4 * (0.3 - self.quest_novelty)
            
        # Severe penalty for infeasible quests
        if self.quest_feasibility < 0.3:
            penalties += 0.6 * (0.3 - self.quest_feasibility)
            
        # Final reward with penalties applied
        reward = base_reward - penalties
        
        # Ensure reward is within reasonable bounds
        reward = max(-1.0, min(1.0, reward))
        
        return reward
