import random
import numpy as np
import json
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class QuestModel:
    """
    Model for representing and generating quest structures.
    
    This class handles the conversion between RL agent actions and
    actual quest structures with objectives, rewards, and narrative elements.
    """
    
    def __init__(self):
        """Initialize the quest model with template data."""
        # Quest type templates
        self.quest_types = [
            "fetch",       # 0: Retrieve an item and bring it back
            "kill",        # 1: Defeat enemies or bosses
            "escort",      # 2: Escort an NPC safely to a destination
            "discover",    # 3: Explore and find locations
            "crafting"     # 4: Gather resources and craft items
        ]
        
        # Templates for quest objectives by type
        self.objective_templates = {
            "fetch": [
                "Find {item} at {location}",
                "Retrieve {item} from {npc}",
                "Collect {amount} {item}s scattered across {location}",
                "Steal {item} from {npc} at {location}",
                "Recover the lost {item} from {location}"
            ],
            "kill": [
                "Defeat {amount} {enemy} at {location}",
                "Slay the {enemy} boss at {location}",
                "Eliminate {npc}'s rival {enemy}",
                "Clear {location} of {enemy} infestation",
                "Ambush and defeat {enemy} patrol near {location}"
            ],
            "escort": [
                "Escort {npc} safely to {location}",
                "Protect {npc} while they {action} at {location}",
                "Guide {npc} through {location} avoiding {enemy}",
                "Ensure {npc} delivers {item} to {location}",
                "Help {npc} escape from {location}"
            ],
            "discover": [
                "Explore the unknown {location}",
                "Map out the area around {location}",
                "Find a path through {location}",
                "Investigate strange occurrences at {location}",
                "Uncover the secret entrance to {location}"
            ],
            "crafting": [
                "Gather {amount} {resource} from {location}",
                "Craft a {item} using resources from {location}",
                "Build a {structure} at {location}",
                "Improve the {item} with materials from {location}",
                "Create {amount} {item}s for {npc}"
            ]
        }
        
        # NPC templates
        self.npc_types = [
            "merchant", "warrior", "sage", "villager", "noble",
            "guard", "outlaw", "wizard", "priest", "traveler"
        ]
        
        # Location templates by theme
        self.location_templates = {
            "fantasy": [
                "ancient ruins", "dark forest", "mountain peak", "underground cavern",
                "enchanted grove", "forgotten temple", "dragon's lair", "goblin camp",
                "elven city", "dwarven mines", "wizard's tower", "haunted castle"
            ],
            "sci-fi": [
                "abandoned space station", "alien planet", "research facility",
                "cyberpunk city", "asteroid belt", "robot graveyard", "quantum lab",
                "holographic simulation", "interstellar colony", "time anomaly"
            ],
            "western": [
                "dusty saloon", "abandoned mine", "canyon pass", "frontier town",
                "railway station", "bandit hideout", "desert oasis", "cattle ranch",
                "ghost town", "mountain trail", "sheriff's office", "native camp"
            ]
        }
        
        # Item templates by theme
        self.item_templates = {
            "fantasy": [
                "ancient scroll", "magic amulet", "enchanted sword", "healing potion",
                "mystical herb", "dragon scale", "elven bow", "wizard's staff",
                "dwarven hammer", "cursed ring", "crystal orb", "phoenix feather"
            ],
            "sci-fi": [
                "data chip", "plasma core", "quantum stabilizer", "neural implant",
                "alien artifact", "encryption key", "antimatter capsule", "holographic map",
                "bionic part", "energy cell", "nanite container", "AI module"
            ],
            "western": [
                "old revolver", "sheriff's badge", "wanted poster", "gold nugget",
                "dynamite stick", "treasure map", "canteen", "lasso",
                "railroad spike", "horseshoe", "medicine bag", "deed to land"
            ]
        }
        
        # Enemy templates by theme
        self.enemy_templates = {
            "fantasy": [
                "goblin", "troll", "dragon", "undead", "evil sorcerer",
                "giant spider", "werewolf", "bandit", "orc", "demon",
                "skeleton warrior", "dark elf"
            ],
            "sci-fi": [
                "rogue AI", "alien organism", "security drone", "mutant",
                "cyborg assassin", "space pirate", "hivemind swarm", "experimental weapon",
                "corrupted android", "parasitic lifeform", "clone soldier", "cosmic entity"
            ],
            "western": [
                "outlaw", "bandit gang", "renegade native", "corrupt sheriff",
                "wild animal", "rival rancher", "train robber", "gunslinger",
                "bounty hunter", "cattle rustler", "dynamite specialist", "desert raider"
            ]
        }
        
        # Resource templates by theme
        self.resource_templates = {
            "fantasy": [
                "mana crystal", "enchanted wood", "dragon bone", "fairy dust",
                "moonstone", "phoenix ash", "troll hide", "mithril ore",
                "elven silk", "magical herb", "goblin steel", "arcane essence"
            ],
            "sci-fi": [
                "rare isotope", "alien alloy", "quantum particle", "synthetic fiber",
                "neural tissue", "superconductor", "exotic gas", "bio-organic compound",
                "nanomaterial", "fusion cell", "data fragment", "dimensional matter"
            ],
            "western": [
                "iron ore", "timber", "leather", "gold dust",
                "silver nugget", "oil", "cotton", "cattle hide",
                "gunpowder", "copper", "medicinal herb", "tobacco leaf"
            ]
        }
        
        # Action templates
        self.action_templates = [
            "investigating", "trading", "negotiating", "performing a ritual",
            "repairing equipment", "gathering intelligence", "healing the wounded",
            "solving a puzzle", "decoding a message", "setting up camp"
        ]
        
        # Default theme
        self.default_theme = "fantasy"
        
        logger.info("Quest model initialized with templates")
        
    def create_quest_from_actions(self, actions, complexity=0.5, theme="fantasy"):
        """
        Create a complete quest structure based on RL agent actions.
        
        Args:
            actions: List of action vectors from the RL agent
            complexity: Overall quest complexity (0.0-1.0)
            theme: Quest theme to use for templates
            
        Returns:
            Complete quest structure as a dictionary
        """
        if not actions:
            raise ValueError("No actions provided for quest generation")
            
        # Use last action for final quest structure
        final_action = actions[-1]
        
        # Extract quest parameters from the action
        quest_type_idx = final_action[0]
        num_objectives = final_action[1] + 1  # 1-5 objectives
        difficulty = final_action[2] / 4.0  # Normalize to 0-1
        reward_magnitude = final_action[3] / 4.0  # Normalize to 0-1
        location_preference = final_action[4]  # Location type preference
        npc_involvement = final_action[5]  # NPC involvement level
        
        # Get quest type
        quest_type = self.quest_types[min(quest_type_idx, len(self.quest_types)-1)]
        
        # Generate appropriate quest title
        title = self._generate_title(quest_type, theme)
        
        # Generate quest description
        description = self._generate_description(quest_type, difficulty, theme)
        
        # Generate objectives
        objectives = self._generate_objectives(
            quest_type, 
            num_objectives, 
            theme,
            difficulty
        )
        
        # Generate rewards
        rewards = self._generate_rewards(reward_magnitude, quest_type, theme)
        
        # Create quest structure
        quest = {
            "id": f"quest_{datetime.now().strftime('%Y%m%d%H%M%S')}_{random.randint(1000, 9999)}",
            "title": title,
            "type": quest_type,
            "theme": theme,
            "description": description,
            "objectives": objectives,
            "rewards": rewards,
            "difficulty": difficulty,
            "complexity": complexity,
            "npc_involvement": npc_involvement / 4.0,  # Normalize to 0-1
            "location_preference": location_preference / 4.0,  # Normalize to 0-1
            "created_at": datetime.now().isoformat()
        }
        
        return quest
    
    def convert_quest_to_actions(self, quest_data):
        """
        Convert a quest structure back to RL actions.
        Useful for evaluating existing quests.
        
        Args:
            quest_data: Quest data structure
            
        Returns:
            List of action vectors
        """
        try:
            # Map quest type to index
            quest_type_idx = self.quest_types.index(quest_data.get("type", "fetch"))
        except ValueError:
            # Default to fetch if type not found
            quest_type_idx = 0
            
        # Map other quest parameters to action values
        num_objectives = len(quest_data.get("objectives", [])) - 1  # 0-4 in action space
        num_objectives = max(0, min(4, num_objectives))  # Constrain to action space
        
        difficulty = quest_data.get("difficulty", 0.5) * 4  # Scale to 0-4
        difficulty = int(max(0, min(4, difficulty)))  # Constrain to action space
        
        # Estimate reward magnitude from rewards if available
        rewards = quest_data.get("rewards", [])
        if rewards and isinstance(rewards, list) and len(rewards) > 0:
            # Assuming rewards have some numeric value to calculate magnitude
            reward_values = [r.get("value", 0) for r in rewards if isinstance(r, dict)]
            if reward_values:
                reward_magnitude = sum(reward_values) / len(reward_values) * 4  # Scale to 0-4
                reward_magnitude = int(max(0, min(4, reward_magnitude)))
            else:
                reward_magnitude = 2  # Default middle value
        else:
            reward_magnitude = 2  # Default middle value
            
        location_preference = int(quest_data.get("location_preference", 0.5) * 4)
        location_preference = max(0, min(4, location_preference))
        
        npc_involvement = int(quest_data.get("npc_involvement", 0.5) * 4)
        npc_involvement = max(0, min(4, npc_involvement))
        
        # Create action vector
        action = [
            quest_type_idx,
            num_objectives,
            difficulty,
            reward_magnitude,
            location_preference,
            npc_involvement
        ]
        
        # We need to return a list of actions, but we only have one aggregate action
        # In a real system, we might reconstruct the sequence of actions
        return [action]
    
    def _generate_title(self, quest_type, theme):
        """Generate an appropriate title for the quest."""
        # Title templates by quest type
        title_templates = {
            "fetch": [
                "The Search for {item}",
                "Lost {item}",
                "Retrieving the {item}",
                "{npc}'s Request",
                "The Missing {item}"
            ],
            "kill": [
                "Hunting the {enemy}",
                "The {enemy} Threat",
                "Eliminate the {enemy}",
                "{location} Cleansing",
                "The {enemy} Problem"
            ],
            "escort": [
                "Protecting {npc}",
                "Safe Passage to {location}",
                "The {npc}'s Journey",
                "Escort Mission",
                "Guarding {npc}"
            ],
            "discover": [
                "Exploring {location}",
                "The Mystery of {location}",
                "Uncharted {location}",
                "Secrets of {location}",
                "The Hidden {location}"
            ],
            "crafting": [
                "Crafting the {item}",
                "{resource} Gathering",
                "The Master Craftsman",
                "Building for {npc}",
                "The {item} Blueprint"
            ]
        }
        
        # Select a random title template for the quest type
        templates = title_templates.get(quest_type, title_templates["fetch"])
        template = random.choice(templates)
        
        # Fill in template with appropriate values
        if "{item}" in template:
            item = random.choice(self.item_templates.get(theme, self.item_templates["fantasy"]))
            template = template.replace("{item}", item)
            
        if "{enemy}" in template:
            enemy = random.choice(self.enemy_templates.get(theme, self.enemy_templates["fantasy"]))
            template = template.replace("{enemy}", enemy)
            
        if "{location}" in template:
            location = random.choice(self.location_templates.get(theme, self.location_templates["fantasy"]))
            template = template.replace("{location}", location)
            
        if "{npc}" in template:
            npc_type = random.choice(self.npc_types)
            template = template.replace("{npc}", f"the {npc_type}")
            
        if "{resource}" in template:
            resource = random.choice(self.resource_templates.get(theme, self.resource_templates["fantasy"]))
            template = template.replace("{resource}", resource)
            
        return template
    
    def _generate_description(self, quest_type, difficulty, theme):
        """Generate a description for the quest based on type and difficulty."""
        # Description templates by quest type
        description_templates = {
            "fetch": [
                "A valuable {item} has been lost in {location}. Your task is to retrieve it and bring it back safely.",
                "{npc} needs you to find a rare {item} from {location} for an important purpose.",
                "The {item} holds great power and must be recovered from {location} before it falls into the wrong hands.",
                "A collection of {item}s must be gathered from across {location} for {npc}'s research."
            ],
            "kill": [
                "The {enemy} have been terrorizing {location}. Defeat them to restore peace.",
                "A powerful {enemy} boss has appeared in {location}. You must defeat it before it grows too strong.",
                "{npc} has tasked you with eliminating the {enemy} threat at {location}.",
                "Clear out the {enemy} infestation that has taken over {location}."
            ],
            "escort": [
                "Guide {npc} safely through the dangerous {location} to their destination.",
                "{npc} has valuable information but is being hunted. Protect them on their journey to {location}.",
                "Ensure {npc} can safely deliver the {item} to {location} without being intercepted by {enemy}.",
                "{npc} needs protection while they perform a critical {action} at {location}."
            ],
            "discover": [
                "A mysterious {location} has been discovered. Explore it and uncover its secrets.",
                "Map out the uncharted {location} and report your findings.",
                "Strange occurrences have been reported at {location}. Investigate the cause.",
                "Find a safe route through {location} for future travelers."
            ],
            "crafting": [
                "Gather rare {resource}s from {location} to craft a powerful {item}.",
                "{npc} needs you to build a {structure} at {location} using locally sourced materials.",
                "Create a special {item} that can help against the {enemy} threat.",
                "Improve the settlement at {location} by crafting essential {item}s for the residents."
            ]
        }
        
        # Difficulty modifiers to add to description
        difficulty_modifiers = {
            (0.0, 0.3): "This should be a straightforward task for someone of your skills.",
            (0.3, 0.6): "The task presents some challenges, but nothing you can't handle.",
            (0.6, 0.8): "This will be a difficult undertaking. Prepare accordingly.",
            (0.8, 1.0): "Only the most skilled adventurers should attempt this extremely dangerous quest."
        }
        
        # Select a difficulty modifier
        difficulty_text = ""
        for (lower, upper), text in difficulty_modifiers.items():
            if lower <= difficulty < upper:
                difficulty_text = text
                break
                
        # Select a random description template for the quest type
        templates = description_templates.get(quest_type, description_templates["fetch"])
        template = random.choice(templates)
        
        # Fill in template with appropriate values
        if "{item}" in template:
            item = random.choice(self.item_templates.get(theme, self.item_templates["fantasy"]))
            template = template.replace("{item}", item)
            
        if "{enemy}" in template:
            enemy = random.choice(self.enemy_templates.get(theme, self.enemy_templates["fantasy"]))
            template = template.replace("{enemy}", enemy)
            
        if "{location}" in template:
            location = random.choice(self.location_templates.get(theme, self.location_templates["fantasy"]))
            template = template.replace("{location}", location)
            
        if "{npc}" in template:
            npc_type = random.choice(self.npc_types)
            template = template.replace("{npc}", f"the {npc_type}")
            
        if "{resource}" in template:
            resource = random.choice(self.resource_templates.get(theme, self.resource_templates["fantasy"]))
            template = template.replace("{resource}", resource)
            
        if "{action}" in template:
            action = random.choice(self.action_templates)
            template = template.replace("{action}", action)
            
        if "{structure}" in template:
            structures = ["shelter", "outpost", "bridge", "tower", "wall", "workshop"]
            structure = random.choice(structures)
            template = template.replace("{structure}", structure)
            
        # Combine description with difficulty text
        full_description = f"{template} {difficulty_text}"
        
        return full_description
    
    def _generate_objectives(self, quest_type, num_objectives, theme, difficulty):
        """Generate quest objectives based on type and number required."""
        objectives = []
        
        # Get appropriate objective templates
        templates = self.objective_templates.get(quest_type, [])
        if not templates:
            logger.warning(f"No objective templates found for quest type '{quest_type}'")
            return objectives
            
        # Ensure we don't request more objectives than we have templates
        num_objectives = min(num_objectives, len(templates))
        
        # Select random templates without replacement
        selected_templates = random.sample(templates, num_objectives)
        
        # Generate each objective
        for i, template in enumerate(selected_templates):
            objective = {
                "id": i + 1,
                "description": self._fill_objective_template(template, theme),
                "completed": False,
                "optional": i >= num_objectives - 1 and num_objectives > 1 and random.random() < 0.3  # Last objective might be optional
            }
            
            # Add difficulty-based requirements
            if "amount" in template:
                # Higher difficulty = more items to collect/enemies to defeat
                base_amount = 3
                amount_modifier = int(difficulty * 10) + 1
                objective["amount_required"] = base_amount + amount_modifier
                objective["amount_completed"] = 0
                
            objectives.append(objective)
            
        return objectives
    
    def _fill_objective_template(self, template, theme):
        """Fill in placeholders in an objective template."""
        result = template
        
        if "{item}" in result:
            item = random.choice(self.item_templates.get(theme, self.item_templates["fantasy"]))
            result = result.replace("{item}", item)
            
        if "{enemy}" in result:
            enemy = random.choice(self.enemy_templates.get(theme, self.enemy_templates["fantasy"]))
            result = result.replace("{enemy}", enemy)
            
        if "{location}" in result:
            location = random.choice(self.location_templates.get(theme, self.location_templates["fantasy"]))
            result = result.replace("{location}", location)
            
        if "{npc}" in result:
            npc_type = random.choice(self.npc_types)
            result = result.replace("{npc}", f"the {npc_type}")
            
        if "{resource}" in result:
            resource = random.choice(self.resource_templates.get(theme, self.resource_templates["fantasy"]))
            result = result.replace("{resource}", resource)
            
        if "{action}" in result:
            action = random.choice(self.action_templates)
            result = result.replace("{action}", action)
            
        if "{structure}" in result:
            structures = ["shelter", "outpost", "bridge", "tower", "wall", "workshop"]
            structure = random.choice(structures)
            result = result.replace("{structure}", structure)
            
        if "{amount}" in result:
            # Random amount between 2 and 10
            amount = random.randint(2, 10)
            result = result.replace("{amount}", str(amount))
            
        return result
    
    def _generate_rewards(self, reward_magnitude, quest_type, theme):
        """Generate appropriate rewards based on quest type and reward magnitude."""
        # Number of rewards depends on magnitude
        num_rewards = max(1, int(reward_magnitude * 3) + 1)
        
        # Base reward value depends on magnitude (scale 10-1000)
        base_value = 10 + int(reward_magnitude * 990)
        
        rewards = []
        
        # Always include some currency reward
        currency_types = {
            "fantasy": "gold coins",
            "sci-fi": "credits",
            "western": "dollars"
        }
        currency = currency_types.get(theme, "gold")
        
        # Currency amount depends on magnitude
        currency_amount = base_value
        
        rewards.append({
            "type": "currency",
            "description": f"{currency_amount} {currency}",
            "value": currency_amount
        })
        
        # Add experience points
        xp_amount = int(base_value * 2)
        rewards.append({
            "type": "experience",
            "description": f"{xp_amount} experience points",
            "value": xp_amount
        })
        
        # Add item rewards based on quest type and theme
        item_reward_templates = {
            "fetch": [
                "{item} of quality",
                "rare {item}",
                "valuable {item}"
            ],
            "kill": [
                "{enemy}'s {item}",
                "looted {item}",
                "battle-tested {item}"
            ],
            "escort": [
                "{npc}'s {item}",
                "protective {item}",
                "traveler's {item}"
            ],
            "discover": [
                "ancient {item}",
                "mysterious {item}",
                "undiscovered {item}"
            ],
            "crafting": [
                "crafted {item}",
                "reinforced {item}",
                "masterwork {item}"
            ]
        }
        
        # Add additional rewards if needed
        if num_rewards > 2:
            # Select templates for this quest type
            templates = item_reward_templates.get(quest_type, item_reward_templates["fetch"])
            
            # Add 1-3 item rewards depending on magnitude
            for i in range(min(3, num_rewards - 2)):
                template = random.choice(templates)
                
                # Fill template
                item_desc = template
                if "{item}" in item_desc:
                    item = random.choice(self.item_templates.get(theme, self.item_templates["fantasy"]))
                    item_desc = item_desc.replace("{item}", item)
                    
                if "{enemy}" in item_desc:
                    enemy = random.choice(self.enemy_templates.get(theme, self.enemy_templates["fantasy"]))
                    item_desc = item_desc.replace("{enemy}", enemy)
                    
                if "{npc}" in item_desc:
                    npc_type = random.choice(self.npc_types)
                    item_desc = item_desc.replace("{npc}", f"{npc_type}")
                
                # Item value is based on base value and random factor
                item_value = int(base_value * (0.5 + random.random()))
                
                rewards.append({
                    "type": "item",
                    "description": item_desc,
                    "value": item_value
                })
        
        return rewards
