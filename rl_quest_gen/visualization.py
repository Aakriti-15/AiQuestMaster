import os
import matplotlib.pyplot as plt
import numpy as np
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

def plot_training_progress(training_stats, save_path='static/plots'):
    """
    Create and save visualizations of training progress.
    
    Args:
        training_stats: Dictionary with training metrics lists
        save_path: Directory to save plots
        
    Returns:
        Path to the generated plot file
    """
    # Create directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)
    
    # Generate timestamp for unique filenames
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    
    try:
        # Create figure with subplots
        fig, axs = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Quest Generation Training Progress', fontsize=16)
        
        # Plot rewards
        rewards = np.array(training_stats.get('rewards', []))
        if len(rewards) > 0:
            episodes = np.arange(1, len(rewards) + 1)
            axs[0, 0].plot(episodes, rewards, 'b-')
            axs[0, 0].set_title('Rewards per Episode')
            axs[0, 0].set_xlabel('Episode')
            axs[0, 0].set_ylabel('Reward')
            
            # Add smoothed reward line
            if len(rewards) >= 10:
                window_size = min(10, len(rewards) // 5)
                smoothed = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
                valid_episodes = episodes[window_size-1:][:len(smoothed)]
                axs[0, 0].plot(valid_episodes, smoothed, 'r-', linewidth=2, label='Moving Average')
                axs[0, 0].legend()
        
        # Plot quest complexity
        complexity = np.array(training_stats.get('quest_complexity', []))
        if len(complexity) > 0:
            episodes = np.arange(1, len(complexity) + 1)
            axs[0, 1].plot(episodes, complexity, 'g-')
            axs[0, 1].set_title('Quest Complexity')
            axs[0, 1].set_xlabel('Episode')
            axs[0, 1].set_ylabel('Complexity')
            axs[0, 1].set_ylim([0, 1])
        
        # Plot quest coherence
        coherence = np.array(training_stats.get('quest_coherence', []))
        if len(coherence) > 0:
            episodes = np.arange(1, len(coherence) + 1)
            axs[1, 0].plot(episodes, coherence, 'm-')
            axs[1, 0].set_title('Quest Coherence')
            axs[1, 0].set_xlabel('Episode')
            axs[1, 0].set_ylabel('Coherence')
            axs[1, 0].set_ylim([0, 1])
        
        # Plot quest novelty
        novelty = np.array(training_stats.get('quest_novelty', []))
        if len(novelty) > 0:
            episodes = np.arange(1, len(novelty) + 1)
            axs[1, 1].plot(episodes, novelty, 'c-')
            axs[1, 1].set_title('Quest Novelty')
            axs[1, 1].set_xlabel('Episode')
            axs[1, 1].set_ylabel('Novelty')
            axs[1, 1].set_ylim([0, 1])
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # Save plot
        plot_filename = f"training_progress_{timestamp}.png"
        plot_path = os.path.join(save_path, plot_filename)
        plt.savefig(plot_path)
        plt.close()
        
        logger.info(f"Training progress plot saved to {plot_path}")
        
        # Return relative path for web display
        return os.path.join(os.path.basename(save_path), plot_filename)
        
    except Exception as e:
        logger.error(f"Error generating training plot: {str(e)}")
        return None

def plot_quest_metrics(quests, save_path='static/plots'):
    """
    Create visualization of generated quest metrics.
    
    Args:
        quests: List of quest dictionaries with metrics
        save_path: Directory to save plots
        
    Returns:
        Path to the generated plot file
    """
    if not quests:
        logger.warning("No quests provided for metric visualization")
        return None
        
    # Create directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)
    
    # Generate timestamp for unique filenames
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    
    try:
        # Extract metrics
        complexities = []
        coherences = []
        novelties = []
        feasibilities = []
        quest_types = []
        
        for quest in quests:
            metrics = quest.get('metrics', {})
            complexities.append(metrics.get('complexity', 0))
            coherences.append(metrics.get('coherence', 0))
            novelties.append(metrics.get('novelty', 0))
            feasibilities.append(metrics.get('feasibility', 0))
            quest_types.append(quest.get('type', 'unknown'))
        
        # Create figure
        fig, axs = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Generated Quest Metrics', fontsize=16)
        
        # Plot metrics distributions
        axs[0, 0].hist(complexities, bins=10, alpha=0.7, color='blue')
        axs[0, 0].set_title('Quest Complexity Distribution')
        axs[0, 0].set_xlabel('Complexity')
        axs[0, 0].set_ylabel('Count')
        
        axs[0, 1].hist(coherences, bins=10, alpha=0.7, color='green')
        axs[0, 1].set_title('Quest Coherence Distribution')
        axs[0, 1].set_xlabel('Coherence')
        axs[0, 1].set_ylabel('Count')
        
        axs[1, 0].hist(novelties, bins=10, alpha=0.7, color='purple')
        axs[1, 0].set_title('Quest Novelty Distribution')
        axs[1, 0].set_xlabel('Novelty')
        axs[1, 0].set_ylabel('Count')
        
        # Count quest types
        type_counts = {}
        for qtype in quest_types:
            type_counts[qtype] = type_counts.get(qtype, 0) + 1
        
        # Plot quest type distribution
        types = list(type_counts.keys())
        counts = [type_counts[t] for t in types]
        axs[1, 1].bar(types, counts, color='orange')
        axs[1, 1].set_title('Quest Type Distribution')
        axs[1, 1].set_xlabel('Quest Type')
        axs[1, 1].set_ylabel('Count')
        plt.setp(axs[1, 1].xaxis.get_majorticklabels(), rotation=45, ha="right")
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # Save plot
        plot_filename = f"quest_metrics_{timestamp}.png"
        plot_path = os.path.join(save_path, plot_filename)
        plt.savefig(plot_path)
        plt.close()
        
        logger.info(f"Quest metrics plot saved to {plot_path}")
        
        # Return relative path for web display
        return os.path.join(os.path.basename(save_path), plot_filename)
        
    except Exception as e:
        logger.error(f"Error generating quest metrics plot: {str(e)}")
        return None

def plot_single_quest_radar(quest, save_path='static/plots'):
    """
    Create a radar chart visualization for a single quest's metrics.
    
    Args:
        quest: Quest dictionary with metrics
        save_path: Directory to save plots
        
    Returns:
        Path to the generated plot file
    """
    if not quest:
        logger.warning("No quest provided for radar visualization")
        return None
        
    # Create directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)
    
    # Generate timestamp for unique filenames
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    quest_id = quest.get('id', 'unknown')
    
    try:
        # Extract metrics
        metrics = quest.get('metrics', {})
        labels = ['Complexity', 'Coherence', 'Novelty', 'Feasibility', 'Reward']
        values = [
            metrics.get('complexity', 0),
            metrics.get('coherence', 0),
            metrics.get('novelty', 0),
            metrics.get('feasibility', 0),
            metrics.get('total_reward', 0) / 2 + 0.5  # Normalize from [-1,1] to [0,1]
        ]
        
        # Complete the loop for the radar chart
        values.append(values[0])
        labels.append(labels[0])
        
        # Convert to radians
        angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
        angles += angles[:1]  # Close the loop
        
        # Create figure
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
        
        # Plot data
        ax.plot(angles, values, 'o-', linewidth=2, label='Metrics')
        ax.fill(angles, values, alpha=0.25)
        
        # Set labels
        ax.set_thetagrids(np.degrees(angles[:-1]), labels[:-1])
        
        # Set radial limits
        ax.set_ylim(0, 1)
        
        # Add title
        plt.title(f"Quest Metrics: {quest.get('title', 'Unnamed Quest')}", size=15)
        
        # Save plot
        plot_filename = f"quest_radar_{quest_id}_{timestamp}.png"
        plot_path = os.path.join(save_path, plot_filename)
        plt.savefig(plot_path)
        plt.close()
        
        logger.info(f"Quest radar plot saved to {plot_path}")
        
        # Return relative path for web display
        return os.path.join(os.path.basename(save_path), plot_filename)
        
    except Exception as e:
        logger.error(f"Error generating quest radar plot: {str(e)}")
        return None
