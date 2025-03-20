import os
import json
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

# Directory for storing generated quests
QUEST_DIR = "data/quests"
# File for storing training statistics
STATS_FILE = "data/training_stats.json"

def ensure_data_dirs():
    """Ensure all required data directories exist."""
    os.makedirs(QUEST_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(STATS_FILE), exist_ok=True)

# Initialize directories
ensure_data_dirs()

def save_quest(quest_data):
    """
    Save a generated quest to file.
    
    Args:
        quest_data: Quest data dictionary
        
    Returns:
        Quest ID string
    """
    # Generate ID if not present
    if "id" not in quest_data:
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        quest_data["id"] = f"quest_{timestamp}"
    
    # Add timestamp if not present
    if "created_at" not in quest_data:
        quest_data["created_at"] = datetime.now().isoformat()
    
    # Save to file
    quest_id = quest_data["id"]
    quest_file = os.path.join(QUEST_DIR, f"{quest_id}.json")
    
    try:
        with open(quest_file, 'w') as f:
            json.dump(quest_data, f, indent=2)
        logger.info(f"Quest saved to {quest_file}")
        return quest_id
    except Exception as e:
        logger.error(f"Error saving quest: {str(e)}")
        return None

def load_quest(quest_id):
    """
    Load a quest from file.
    
    Args:
        quest_id: ID of the quest to load
        
    Returns:
        Quest data dictionary or None if not found
    """
    quest_file = os.path.join(QUEST_DIR, f"{quest_id}.json")
    
    try:
        if os.path.exists(quest_file):
            with open(quest_file, 'r') as f:
                return json.load(f)
        else:
            logger.warning(f"Quest file not found: {quest_file}")
            return None
    except Exception as e:
        logger.error(f"Error loading quest {quest_id}: {str(e)}")
        return None

def load_quests(limit=None):
    """
    Load all saved quests.
    
    Args:
        limit: Maximum number of quests to load (most recent first)
        
    Returns:
        List of quest data dictionaries
    """
    ensure_data_dirs()
    quests = []
    
    try:
        # Get all JSON files in quest directory
        quest_files = [f for f in os.listdir(QUEST_DIR) if f.endswith('.json')]
        
        # Sort by modification time (newest first)
        quest_files.sort(key=lambda f: os.path.getmtime(os.path.join(QUEST_DIR, f)), reverse=True)
        
        # Apply limit if specified
        if limit is not None:
            quest_files = quest_files[:limit]
        
        # Load each quest
        for filename in quest_files:
            try:
                with open(os.path.join(QUEST_DIR, filename), 'r') as f:
                    quest_data = json.load(f)
                    quests.append(quest_data)
            except Exception as e:
                logger.error(f"Error loading quest file {filename}: {str(e)}")
                continue
                
        logger.info(f"Loaded {len(quests)} quests")
        return quests
    except Exception as e:
        logger.error(f"Error loading quests: {str(e)}")
        return []

def delete_quest(quest_id):
    """
    Delete a quest file.
    
    Args:
        quest_id: ID of the quest to delete
        
    Returns:
        True if successful, False otherwise
    """
    quest_file = os.path.join(QUEST_DIR, f"{quest_id}.json")
    
    try:
        if os.path.exists(quest_file):
            os.remove(quest_file)
            logger.info(f"Deleted quest {quest_id}")
            return True
        else:
            logger.warning(f"Quest file not found for deletion: {quest_file}")
            return False
    except Exception as e:
        logger.error(f"Error deleting quest {quest_id}: {str(e)}")
        return False

def save_training_stats(stats):
    """
    Save training statistics to file.
    
    Args:
        stats: Training statistics dictionary
        
    Returns:
        True if successful, False otherwise
    """
    ensure_data_dirs()
    
    try:
        # Add timestamp
        stats_with_time = stats.copy()
        stats_with_time["updated_at"] = datetime.now().isoformat()
        
        with open(STATS_FILE, 'w') as f:
            json.dump(stats_with_time, f, indent=2)
        logger.info(f"Training stats saved to {STATS_FILE}")
        return True
    except Exception as e:
        logger.error(f"Error saving training stats: {str(e)}")
        return False

def load_training_stats():
    """
    Load training statistics from file.
    
    Returns:
        Training statistics dictionary or empty dict if not found
    """
    ensure_data_dirs()
    
    try:
        if os.path.exists(STATS_FILE):
            with open(STATS_FILE, 'r') as f:
                return json.load(f)
        else:
            logger.info(f"No training stats file found at {STATS_FILE}")
            return {}
    except Exception as e:
        logger.error(f"Error loading training stats: {str(e)}")
        return {}

def export_quests_to_json(quests, output_file):
    """
    Export a collection of quests to a single JSON file.
    
    Args:
        quests: List of quest dictionaries
        output_file: Path to output file
        
    Returns:
        True if successful, False otherwise
    """
    try:
        with open(output_file, 'w') as f:
            json.dump(quests, f, indent=2)
        logger.info(f"Exported {len(quests)} quests to {output_file}")
        return True
    except Exception as e:
        logger.error(f"Error exporting quests: {str(e)}")
        return False

def import_quests_from_json(input_file):
    """
    Import quests from a JSON file and save them individually.
    
    Args:
        input_file: Path to input file containing quest data
        
    Returns:
        Number of quests imported
    """
    try:
        with open(input_file, 'r') as f:
            quests_data = json.load(f)
            
        if not isinstance(quests_data, list):
            logger.error(f"Invalid quest data format in {input_file}")
            return 0
            
        import_count = 0
        for quest in quests_data:
            if save_quest(quest):
                import_count += 1
                
        logger.info(f"Imported {import_count} quests from {input_file}")
        return import_count
    except Exception as e:
        logger.error(f"Error importing quests: {str(e)}")
        return 0
