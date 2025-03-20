import numpy as np
import logging

logger = logging.getLogger(__name__)

class QuestRewardFunction:
    """
    Reward function for evaluating quest quality.
    
    This class implements various metrics to measure the quality of generated
    quests, including complexity, coherence, novelty, and player engagement.
    """
    
    def __init__(self, complexity_weight=0.3, coherence_weight=0.3,
                 novelty_weight=0.2, feasibility_weight=0.2):
        """
        Initialize the reward function with weights for different components.
        
        Args:
            complexity_weight: Weight for quest complexity in reward
            coherence_weight: Weight for quest coherence in reward
            novelty_weight: Weight for quest novelty in reward
            feasibility_weight: Weight for quest feasibility in reward
        """
        self.complexity_weight = complexity_weight
        self.coherence_weight = coherence_weight
        self.novelty_weight = novelty_weight
        self.feasibility_weight = feasibility_weight
        
        # Quest history for novelty calculation
        self.quest_history = []
        self.max_history = 100  # Remember last 100 quests
        
        # Preferences for target complexity
        self.target_complexity = 0.6  # Moderately complex quests preferred
        self.complexity_tolerance = 0.2  # Acceptable range around target
        
        logger.info("Quest reward function initialized")
        
    def calculate_reward(self, quest, game_state=None):
        """
        Calculate the reward for a generated quest.
        
        Args:
            quest: The quest structure to evaluate
            game_state: Optional game state for context
            
        Returns:
            Calculated reward value
        """
        if not quest:
            logger.warning("Empty quest provided for reward calculation")
            return -1.0  # Penalty for empty quest
            
        # Calculate individual reward components
        complexity_reward = self._evaluate_complexity(quest)
        coherence_reward = self._evaluate_coherence(quest)
        novelty_reward = self._evaluate_novelty(quest)
        feasibility_reward = self._evaluate_feasibility(quest, game_state)
        
        # Calculate total reward
        total_reward = (
            self.complexity_weight * complexity_reward +
            self.coherence_weight * coherence_reward +
            self.novelty_weight * novelty_reward +
            self.feasibility_weight * feasibility_reward
        )
        
        # Add the quest to history for future novelty calculations
        self._add_to_history(quest)
        
        # Log detailed breakdown
        logger.debug(f"Reward breakdown - Complexity: {complexity_reward:.2f}, "
                   f"Coherence: {coherence_reward:.2f}, Novelty: {novelty_reward:.2f}, "
                   f"Feasibility: {feasibility_reward:.2f}, Total: {total_reward:.2f}")
        
        return total_reward
        
    def _evaluate_complexity(self, quest):
        """
        Evaluate the complexity of a quest.
        
        Higher rewards for quests with complexity close to the target complexity.
        Penalties for too simple or too complex quests.
        
        Args:
            quest: Quest structure to evaluate
            
        Returns:
            Complexity reward component
        """
        # Base complexity on number of objectives, dependencies, and difficulty
        num_objectives = len(quest.get("objectives", []))
        difficulty = quest.get("difficulty", 0.5)
        
        # Calculate raw complexity (0-1 scale)
        raw_complexity = (num_objectives / 10.0) * 0.7 + difficulty * 0.3
        raw_complexity = min(1.0, raw_complexity)  # Cap at 1.0
        
        # Preference for target complexity
        distance_from_target = abs(raw_complexity - self.target_complexity)
        
        # If within tolerance, give high reward
        if distance_from_target <= self.complexity_tolerance:
            # Scale from 0.8-1.0 based on closeness to target
            return 1.0 - (distance_from_target / self.complexity_tolerance) * 0.2
        else:
            # Penalty increases with distance from target
            excess_distance = distance_from_target - self.complexity_tolerance
            return 0.8 - excess_distance * 2.0  # Sharper penalty for being far from target
    
    def _evaluate_coherence(self, quest):
        """
        Evaluate how coherent and logical the quest is.
        
        Higher rewards for quests where objectives, rewards, and theme align.
        
        Args:
            quest: Quest structure to evaluate
            
        Returns:
            Coherence reward component
        """
        # Start with perfect coherence
        coherence = 1.0
        
        # Check if quest has required fields
        required_fields = ["title", "description", "objectives", "rewards", "type"]
        for field in required_fields:
            if field not in quest or not quest[field]:
                coherence -= 0.2  # Penalty for missing required fields
        
        # Check if objectives match quest type
        quest_type = quest.get("type", "")
        objectives = quest.get("objectives", [])
        
        if objectives and quest_type:
            # Count objectives that seem relevant to the quest type
            relevant_count = 0
            for obj in objectives:
                obj_desc = obj.get("description", "").lower()
                
                # Check for keywords related to quest type
                if quest_type == "fetch" and any(word in obj_desc for word in ["find", "retrieve", "collect", "gather", "bring"]):
                    relevant_count += 1
                elif quest_type == "kill" and any(word in obj_desc for word in ["kill", "defeat", "slay", "eliminate", "destroy"]):
                    relevant_count += 1
                elif quest_type == "escort" and any(word in obj_desc for word in ["escort", "protect", "guide", "bring", "accompany"]):
                    relevant_count += 1
                elif quest_type == "discover" and any(word in obj_desc for word in ["find", "discover", "explore", "map", "investigate"]):
                    relevant_count += 1
                elif quest_type == "crafting" and any(word in obj_desc for word in ["craft", "create", "build", "make", "construct"]):
                    relevant_count += 1
            
            # Calculate percentage of relevant objectives
            if len(objectives) > 0:
                relevance_ratio = relevant_count / len(objectives)
                # Penalize if less than 70% of objectives are relevant
                if relevance_ratio < 0.7:
                    coherence -= 0.3 * (0.7 - relevance_ratio)
        
        # Ensure reward doesn't go below 0
        return max(0.0, coherence)
    
    def _evaluate_novelty(self, quest):
        """
        Evaluate how novel the quest is compared to previously seen quests.
        
        Higher rewards for quests that differ from recent quests.
        
        Args:
            quest: Quest structure to evaluate
            
        Returns:
            Novelty reward component
        """
        if not self.quest_history:
            # First quest is automatically novel
            return 1.0
            
        # Calculate similarity scores against previous quests
        similarity_scores = []
        
        for historic_quest in self.quest_history:
            similarity = self._calculate_quest_similarity(quest, historic_quest)
            similarity_scores.append(similarity)
        
        # Calculate novelty as the inverse of the highest similarity
        max_similarity = max(similarity_scores) if similarity_scores else 0
        novelty = 1.0 - max_similarity
        
        return novelty
    
    def _calculate_quest_similarity(self, quest1, quest2):
        """Calculate similarity between two quests."""
        # Compare quest types (if same type, base similarity of 0.3)
        base_similarity = 0.3 if quest1.get("type") == quest2.get("type") else 0.0
        
        # Compare objectives
        objectives1 = [obj.get("description", "") for obj in quest1.get("objectives", [])]
        objectives2 = [obj.get("description", "") for obj in quest2.get("objectives", [])]
        
        # Simple text similarity for objectives
        objective_similarity = 0.0
        if objectives1 and objectives2:
            common_words = set()
            all_words = set()
            
            # Extract words from objectives
            for obj in objectives1:
                words = set(obj.lower().split())
                common_words.update(words)
                all_words.update(words)
            
            # Find common words with second quest
            for obj in objectives2:
                words = set(obj.lower().split())
                common_words &= words  # Intersection
                all_words.update(words)
            
            # Jaccard similarity
            if all_words:
                objective_similarity = len(common_words) / len(all_words)
        
        # Compare difficulty
        difficulty_similarity = 1.0 - abs(quest1.get("difficulty", 0.5) - quest2.get("difficulty", 0.5))
        
        # Calculate overall similarity
        similarity = (0.4 * base_similarity + 
                      0.4 * objective_similarity + 
                      0.2 * difficulty_similarity)
        
        return similarity
    
    def _evaluate_feasibility(self, quest, game_state=None):
        """
        Evaluate if the quest is feasible given the game state.
        
        Higher rewards for quests that can be completed with available resources.
        
        Args:
            quest: Quest structure to evaluate
            game_state: Game state information
            
        Returns:
            Feasibility reward component
        """
        # Start with perfect feasibility
        feasibility = 1.0
        
        # If no game state provided, use default assessment
        if not game_state:
            # Base feasibility on quest difficulty
            difficulty = quest.get("difficulty", 0.5)
            
            # Very difficult quests might be less feasible
            if difficulty > 0.8:
                feasibility -= (difficulty - 0.8) * 2  # Penalty for extremely difficult quests
            
            # Check for optional objectives
            objectives = quest.get("objectives", [])
            required_objectives = [obj for obj in objectives if not obj.get("optional", False)]
            
            # Too many required objectives might reduce feasibility
            if len(required_objectives) > 5:
                feasibility -= 0.1 * (len(required_objectives) - 5)
        else:
            # With game state, we could implement more sophisticated checks
            # This would depend on the specific game implementation
            pass
        
        # Ensure reward doesn't go below 0
        return max(0.0, feasibility)
    
    def _add_to_history(self, quest):
        """Add quest to history for novelty calculation."""
        # Create a simplified version for history to save memory
        simplified = {
            "type": quest.get("type", ""),
            "difficulty": quest.get("difficulty", 0.5),
            "objectives": [obj.get("description", "") for obj in quest.get("objectives", [])]
        }
        
        # Add to history
        self.quest_history.append(simplified)
        
        # Maintain maximum history size
        if len(self.quest_history) > self.max_history:
            self.quest_history.pop(0)  # Remove oldest
