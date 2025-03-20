import os
import logging
from flask import Flask, render_template, jsonify, request, redirect, url_for, flash
import numpy as np
from rl_quest_gen.environment import GameEnvironment
from rl_quest_gen.agent import QuestGenerationAgent
from rl_quest_gen.quest_model import QuestModel
from rl_quest_gen.visualization import plot_training_progress, plot_quest_metrics, plot_single_quest_radar
from utils.data_handler import save_quest, load_quests, load_training_stats, delete_quest, export_quests_to_json

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
    """Home page for the quest generation system."""
    redirect_to_gen = request.args.get('auto_generate', False)
    if redirect_to_gen:
        return redirect(url_for('generate_page'))
    return render_template('index.html')

@app.route('/generate')
def generate_page():
    """Page for generating new quests."""
    # Get any pre-selected parameters from URL
    theme = request.args.get('theme', 'fantasy')
    quest_type = request.args.get('type', None)
    complexity = request.args.get('complexity', 0.5)
    
    try:
        complexity = float(complexity)
    except ValueError:
        complexity = 0.5
        
    return render_template('generate.html', 
                          theme=theme, 
                          quest_type=quest_type, 
                          complexity=complexity)

@app.route('/api/generate_quest', methods=['POST'])
def generate_quest():
    """API endpoint to generate a new quest."""
    try:
        # Get generation parameters
        complexity = float(request.form.get('complexity', 0.5))
        theme = request.form.get('theme', 'fantasy')
        quest_type = request.form.get('quest_type', None)
        deterministic = request.form.get('deterministic', 'true').lower() == 'true'
        save_quest_option = request.form.get('save', 'true').lower() == 'true'
        
        # Set random seed if provided
        seed = request.form.get('seed')
        if seed:
            seed = int(seed)
            np.random.seed(seed)
        
        # Generate quest
        quest = agent.generate_quest(complexity=complexity, theme=theme, deterministic=deterministic)
        
        # If a specific quest type was requested, ensure the quest matches that type
        if quest_type and quest['type'] != quest_type:
            quest['type'] = quest_type
            # Re-generate objectives based on the new type
            objectives = quest_model._generate_objectives(
                quest_type, 
                len(quest['objectives']), 
                theme,
                quest['difficulty']
            )
            quest['objectives'] = objectives
            
            # Re-generate title based on the new type
            quest['title'] = quest_model._generate_title(quest_type, theme)
        
        # Save generated quest if requested
        quest_id = None
        if save_quest_option:
            quest_id = save_quest(quest)
        
        return jsonify({
            'success': True,
            'quest': quest,
            'quest_id': quest_id
        })
    except Exception as e:
        logger.error(f"Quest generation error: {str(e)}")
        return jsonify({'success': False, 'message': f'Error generating quest: {str(e)}'})

@app.route('/my-quests')
def my_quests():
    """Page for viewing saved quests."""
    quests = load_quests()
    return render_template('my_quests.html', quests=quests)

@app.route('/api/delete_quest/<quest_id>', methods=['POST'])
def delete_quest_api(quest_id):
    """API endpoint to delete a quest."""
    try:
        success = delete_quest(quest_id)
        if success:
            return jsonify({'success': True, 'message': 'Quest deleted successfully'})
        else:
            return jsonify({'success': False, 'message': 'Quest not found or could not be deleted'})
    except Exception as e:
        logger.error(f"Quest deletion error: {str(e)}")
        return jsonify({'success': False, 'message': f'Error deleting quest: {str(e)}'})

@app.route('/api/export_quests', methods=['POST'])
def export_quests():
    """API endpoint to export quests to JSON."""
    try:
        quests = load_quests()
        if not quests:
            return jsonify({'success': False, 'message': 'No quests to export'})
            
        export_path = 'static/exports'
        os.makedirs(export_path, exist_ok=True)
        
        timestamp = np.datetime64('now').astype(str).replace(':', '-').replace(' ', '_')
        filename = f"quests_export_{timestamp}.json"
        export_file = os.path.join(export_path, filename)
        
        success = export_quests_to_json(quests, export_file)
        if success:
            return jsonify({
                'success': True, 
                'message': 'Quests exported successfully',
                'download_url': f"/static/exports/{filename}"
            })
        else:
            return jsonify({'success': False, 'message': 'Error exporting quests'})
    except Exception as e:
        logger.error(f"Quest export error: {str(e)}")
        return jsonify({'success': False, 'message': f'Error exporting quests: {str(e)}'})

@app.route('/quest/<quest_id>')
def view_quest(quest_id):
    """Page for viewing a single quest."""
    from utils.data_handler import load_quest
    quest = load_quest(quest_id)
    if not quest:
        flash('Quest not found', 'error')
        return redirect(url_for('my_quests'))
        
    # Generate radar chart
    radar_chart = plot_single_quest_radar(quest)
    
    return render_template('view_quest.html', quest=quest, radar_chart=radar_chart)

# Error handlers
@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

@app.errorhandler(500)
def server_error(e):
    return render_template('500.html'), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
