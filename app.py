import os
import logging
from flask import Flask, render_template, jsonify, request, redirect, url_for, flash
import numpy as np
from rl_quest_gen.environment import GameEnvironment
from rl_quest_gen.agent import QuestGenerationAgent
from rl_quest_gen.quest_model import QuestModel
from rl_quest_gen.visualization import plot_training_progress
from utils.data_handler import save_quest, load_quests, load_training_stats

# Setup logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "dev-secret-key")

# Initialize game environment and RL agent
env = GameEnvironment()
quest_model = QuestModel()
agent = QuestGenerationAgent(env, quest_model)

@app.route('/')
def index():
    """Home page with overview of the quest generation system."""
    return render_template('index.html')

@app.route('/train')
def train_page():
    """Page for training the RL agent."""
    training_stats = load_training_stats()
    return render_template('train.html', training_stats=training_stats)

@app.route('/api/train', methods=['POST'])
def train_agent():
    """API endpoint to start the training process."""
    try:
        # Get training parameters from form
        episodes = int(request.form.get('episodes', 100))
        learning_rate = float(request.form.get('learning_rate', 0.001))
        exploration_rate = float(request.form.get('exploration_rate', 0.1))
        
        # Start training process
        training_results = agent.train(
            num_episodes=episodes,
            learning_rate=learning_rate,
            exploration_rate=exploration_rate
        )
        
        # Generate training visualization
        plot_path = plot_training_progress(training_results)
        
        return jsonify({
            'success': True,
            'message': f'Training completed after {episodes} episodes',
            'plot_path': plot_path,
            'training_results': {
                'avg_reward': float(np.mean(training_results['rewards'])),
                'final_reward': float(training_results['rewards'][-1]),
                'quest_complexity': float(np.mean(training_results['quest_complexity']))
            }
        })
    except Exception as e:
        logger.error(f"Training error: {str(e)}")
        return jsonify({'success': False, 'message': f'Error during training: {str(e)}'})

@app.route('/generate')
def generate_page():
    """Page for generating new quests."""
    return render_template('generate.html')

@app.route('/api/generate_quest', methods=['POST'])
def generate_quest():
    """API endpoint to generate a new quest."""
    try:
        # Get generation parameters
        complexity = float(request.form.get('complexity', 0.5))
        theme = request.form.get('theme', 'fantasy')
        seed = request.form.get('seed')
        if seed:
            seed = int(seed)
            np.random.seed(seed)
        
        # Generate quest
        quest = agent.generate_quest(complexity=complexity, theme=theme)
        
        # Save generated quest
        quest_id = save_quest(quest)
        
        return jsonify({
            'success': True,
            'quest': quest,
            'quest_id': quest_id
        })
    except Exception as e:
        logger.error(f"Quest generation error: {str(e)}")
        return jsonify({'success': False, 'message': f'Error generating quest: {str(e)}'})

@app.route('/evaluate')
def evaluate_page():
    """Page for evaluating generated quests."""
    quests = load_quests()
    return render_template('evaluate.html', quests=quests)

@app.route('/api/evaluate_quest', methods=['POST'])
def evaluate_quest():
    """API endpoint to evaluate a quest."""
    try:
        quest_data = request.json.get('quest')
        evaluation = agent.evaluate_quest(quest_data)
        
        return jsonify({
            'success': True,
            'evaluation': evaluation
        })
    except Exception as e:
        logger.error(f"Quest evaluation error: {str(e)}")
        return jsonify({'success': False, 'message': f'Error evaluating quest: {str(e)}'})

# Error handlers
@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

@app.errorhandler(500)
def server_error(e):
    return render_template('500.html'), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
