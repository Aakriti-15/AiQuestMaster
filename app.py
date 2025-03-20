import os
import logging
import json
import random
from datetime import datetime
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
    """Home page - redirects to adventure game."""
    return redirect(url_for('adventure_game'))

@app.route('/adventure')
def adventure_game():
    """Interactive adventure game page."""
    return render_template('adventure_game.html')

@app.route('/api/adventure/generate_scenario', methods=['POST'])
def generate_scenario():
    """Generate a new scenario for the adventure game."""
    try:
        # Get parameters
        theme = request.json.get('theme', 'fantasy')
        previous_choice = request.json.get('previous_choice', None)
        player_state = request.json.get('player_state', {})
        history = request.json.get('history', [])

        # In a production app, this would use our RL model to generate the next scenario
        # For now, we'll use pre-defined templates and adapt them based on inputs
        
        # Generate scenario options based on theme
        scenario_templates = {
            'fantasy': [
                {
                    'text': "You find yourself standing at the entrance of a dark cave. Mysterious sounds echo from within, and you notice strange markings carved into the rocks nearby. A cool breeze flows from the cave, carrying an unfamiliar scent.",
                    'choices': [
                        {"id": "enter_cave", "text": "Enter the cave cautiously"},
                        {"id": "examine_markings", "text": "Examine the strange markings more closely"},
                        {"id": "look_around", "text": "Look around for other paths or signs"}
                    ]
                },
                {
                    'text': "A thick forest surrounds you, sunlight filtering through the dense canopy above. The path ahead splits around an ancient oak tree, its trunk wider than three men standing side by side. There's a small hollow in the tree that seems unnatural.",
                    'choices': [
                        {"id": "take_left_path", "text": "Take the left path deeper into the forest"},
                        {"id": "take_right_path", "text": "Take the right path toward a clearing"},
                        {"id": "examine_tree", "text": "Investigate the hollow in the ancient oak"}
                    ]
                },
                {
                    'text': "The village square is bustling with activity as locals prepare for the harvest festival. An old man with a silver beard watches you from beside the well, his eyes showing recognition though you've never met. Near the tavern, a group of armored guards speak in hushed voices.",
                    'choices': [
                        {"id": "talk_to_old_man", "text": "Approach the old man by the well"},
                        {"id": "listen_to_guards", "text": "Try to overhear what the guards are discussing"},
                        {"id": "enter_tavern", "text": "Enter the tavern to gather information"}
                    ]
                }
            ],
            'sci-fi': [
                {
                    'text': "The space station's warning lights flash red as the emergency siren blares through the corridors. Through the viewport, you see debris from what appears to be a destroyed ship floating in space. Your communication console beeps with an incoming transmission.",
                    'choices': [
                        {"id": "answer_transmission", "text": "Answer the incoming transmission"},
                        {"id": "check_damage", "text": "Check the station's damage reports"},
                        {"id": "prepare_escape", "text": "Prepare the escape pod for possible evacuation"}
                    ]
                },
                {
                    'text': "The neon lights of New Shanghai flicker overhead as rain pours down on the crowded streets. Your augmented reality display highlights a suspicious figure ducking into an alley. Your mission target is somewhere in this district, according to your handler.",
                    'choices': [
                        {"id": "follow_figure", "text": "Follow the suspicious figure into the alley"},
                        {"id": "scan_crowd", "text": "Use enhanced scanner to analyze the crowd"},
                        {"id": "contact_handler", "text": "Contact your handler for updated instructions"}
                    ]
                },
                {
                    'text': "The laboratory door slides open with a soft hiss. Inside, holographic displays show data streams and molecular structures floating in the air. A synthetic assistant looks up from a workstation. 'Authorization required for further access,' it states in a monotone voice.",
                    'choices': [
                        {"id": "show_credentials", "text": "Present your credentials to the synthetic assistant"},
                        {"id": "override_system", "text": "Attempt to override the security system"},
                        {"id": "bluff", "text": "Bluff your way past with confidence and technical jargon"}
                    ]
                }
            ],
            'western': [
                {
                    'text': "The dusty main street of Redemption Creek stretches before you, the afternoon sun casting long shadows from the wooden buildings on either side. From the saloon comes the tinkling of a poorly-tuned piano. Outside the sheriff's office, a notice board displays several wanted posters.",
                    'choices': [
                        {"id": "enter_saloon", "text": "Push through the swinging doors of the saloon"},
                        {"id": "check_wanted", "text": "Examine the wanted posters outside the sheriff's office"},
                        {"id": "visit_general_store", "text": "Head to the general store for supplies"}
                    ]
                },
                {
                    'text': "Your horse's hooves kick up dust from the trail as you approach the canyon. Below, a river snakes through the red rock. In the distance, you spot smoke rising - perhaps a campfire. Your canteen is running low on water.",
                    'choices': [
                        {"id": "investigate_smoke", "text": "Ride toward the smoke to investigate"},
                        {"id": "descend_to_river", "text": "Find a path down to the river to refill your canteen"},
                        {"id": "continue_journey", "text": "Continue on your current path through the canyon"}
                    ]
                },
                {
                    'text': "The mining camp is alive with activity despite the late hour. Rough-looking men gather around fires, sharing bottles and stories. A large tent at the edge of camp seems to serve as some kind of headquarters, with armed guards posted outside.",
                    'choices': [
                        {"id": "join_campfire", "text": "Approach one of the campfires to socialize"},
                        {"id": "approach_tent", "text": "Walk toward the guarded tent to learn more"},
                        {"id": "observe_quietly", "text": "Find a quiet spot to observe the camp before making a move"}
                    ]
                }
            ]
        }
        
        # Select a scenario based on previous choice or randomly if it's the first one
        scenarios = scenario_templates.get(theme, scenario_templates['fantasy'])
        
        if previous_choice:
            # In a real implementation, this would use our RL model to select or generate
            # a scenario based on the player's previous choice
            # For demo, we'll just pick a random one
            scenario = random.choice(scenarios)
        else:
            # For first scenario, pick first in list to ensure consistent starting point
            scenario = scenarios[0]
        
        # Add any consequences based on previous choice
        consequences = {}
        if previous_choice:
            # Risky choices might affect health
            if previous_choice in ['enter_cave', 'follow_figure', 'investigate_smoke']:
                consequences['health_change'] = -5
                consequences['xp_change'] = 10
            else:
                consequences['xp_change'] = 5
            
            # Some choices might add items
            if previous_choice in ['examine_tree', 'check_damage', 'visit_general_store']:
                consequences['new_item'] = {
                    'id': f"item_{random.randint(1000, 9999)}",
                    'name': "Mysterious Object",
                    'description': "Something you found that might be useful later."
                }
        
        return jsonify({
            'success': True,
            'scenario': scenario,
            'consequences': consequences
        })
    except Exception as e:
        logger.error(f"Scenario generation error: {str(e)}")
        return jsonify({'success': False, 'message': f'Error generating scenario: {str(e)}'})

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
