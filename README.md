this is a group project that we made in 6th semester of my Btech college
#  RL Quest Generator

A Flask web application that dynamically generates interactive adventure scenarios and quests using reinforcement learning-inspired logic. Supports multiple themes like **fantasy**, **sci-fi**, and **western**.

![Project Icon](generated-icon.png)

##  Features

-  Themed interactive scenario generation (Fantasy, Sci-Fi, Western)
-  Quest generation using an environment-agent-model pipeline
-  Save, delete, and export custom quests
-  Quest complexity control and metrics visualization
- Clean UI with template-based rendering (HTML templates)
-  RESTful API for frontend/backend interaction

##  Tech Stack

- **Python 3.11+**
- **Flask 3.1**
- **Numpy**
- **Matplotlib**
- **Flask-SQLAlchemy**
- **Gym (for environment logic)**
- **Gunicorn (for deployment)**
- **PostgreSQL** (assumed from psycopg2-binary)
- **Replit/Nix** dev environment

##  Project Structure

```bash
.
├── app.py                  # Flask app with all routes and core logic
├── main.py                 # Entry point (runs the app)
├── replit.nix              # Replit build config (Nix environment)
├── pyproject.toml          # Python dependencies and metadata
├── uv.lock                 # Lockfile for uv package manager
├── static/                 # Exported quest files and assets
├── templates/              # HTML templates for UI
├── rl_quest_gen/           # Core RL logic (agent, environment, quest model)
├── utils/                  # Data handler scripts for saving/loading quests
├── generated-icon.png      # Project icon
```

##  API Endpoints

### `/api/adventure/generate_scenario` `[POST]`
Generates an adventure scenario.
- **Input**: `{ "theme": "fantasy", "previous_choice": "...", "player_state": {}, "history": [] }`
- **Output**: Scenario text and choices with optional consequences

### `/api/generate_quest` `[POST]`
Generates a new quest.
- **Form Params**: `theme`, `complexity`, `quest_type`, `deterministic`, `save`, `seed`
- **Returns**: Quest object

### `/api/delete_quest/<quest_id>` `[POST]`
Deletes a saved quest by ID.

### `/api/export_quests` `[POST]`
Exports saved quests to a downloadable `.json` file.

##  Setup Instructions

1. **Clone the repo**:
   ```bash
   git clone https://github.com/your-username/your-repo-name.git
   cd your-repo-name
   ```

2. **Install dependencies** (using `uv` or `pip`):
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the app**:
   ```bash
   python main.py
   ```
   Or use Gunicorn for production:
   ```bash
   gunicorn main:app
   ```

4. **Access**:
   Open your browser at [http://localhost:5000](http://localhost:5000)


