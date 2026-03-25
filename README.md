# Agro-Hub: Soil-Aware Crop Rotation Optimization

Agro-Hub is a research-driven dashboard that uses Deep Reinforcement Learning (DRL) via Proximal Policy Optimization (PPO) to solve the complex problem of sustainable agriculture. It provides farmers with a "smart advisor" that balances immediate financial profit (using live market data from Yahoo Finance) with long-term soil health and sustainability. 

All predictions, interactions, and historical requests are tracked intelligently using a local SQLite database for easy logging and auditing.

---

## 🛠️ Prerequisites
You need **Python 3.8+** installed on your system.

---

## 🚀 Setup & Installation

Open your terminal/command prompt and navigate to the `Agro_hub` directory.

### 1. Create a Virtual Environment

**Windows:**
```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

**Mac/Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install Dependencies
Once your virtual environment is active (you should see `(.venv)` in your terminal prompt), install the required packages:

```bash
pip install -r requirements.txt
```

*(This will install Flask, PyTorch, Stable-Baselines3, yfinance, and Gymnasium dependencies).*

---

## 🏃‍♂️ Running the Project

### 1. Start the Backend API
Ensure your virtual environment is active, then start the Flask server from the root of the project:

**Windows / Mac / Linux:**
```bash
python scripts/api.py
```
*You should see a message saying the server is running on `http://127.0.0.1:5000`.*

### 2. Open the Frontend Dashboard
The frontend is completely static and does not require a web server like Node.js or React. 
Simply open the `index.html` file in your preferred web browser:
1. Double-click `index.html`.
2. OR Drag and Drop `index.html` into Chrome/Firefox/Edge.

### 3. Using the App
- Input arbitrary numbers in the **Manual Soil Input** fields, or build up a history using the **Crop Type** dropdown.
- Click **Get ML Recommendation**.
- The page will automatically ping the backend API, save the request to the SQLite database, and return the UI results.
- Click the **Load Activity Log** button at the bottom of the page to see a historical record of your AI requests fetched straight from the database.

---

## 🧪 Advanced Execution (Optional)

If you want to view the raw Reinforcement Learning logic or retrain the model, you can run the offline scripts (these work seamlessly across Mac/Win/Linux):

- **Evaluate the Model:** Generates performance comparison charts (PPO vs Greedy) inside the `/results/plots/` folder.
  ```bash
  python scripts/evaluate.py
  ```
- **Retrain the Model:** Runs the simulation environment for 100,000 timesteps and overwrites the `.zip` model file.
  ```bash
  python scripts/train.py
  ```

---

## 📁 Project Structure

```text
Agro_hub/
├── docs/                 # Documentation files
├── env/
│   └── agro_env.py       # Custom Gymnasium Environment simulation
├── frontend/
│   ├── app.js            # Frontend logic & API fetching
│   └── style.css         # Glassmorphism UI styling
├── models/
│   └── ppo_agro_final.zip # The actual pre-trained AI weights
├── results/              # Auto-generated plots and metrics
├── scripts/
│   ├── api.py            # FLASK Backend application & Routing
│   ├── db.py             # SQLite helper and database initializer
│   ├── evaluate.py       # Analytics plotter script
│   └── train.py          # RL Training script
├── index.html            # Main Dashboard UI
├── database.db           # Auto-generated SQLite history tracking
├── novelty.txt           # Academic write-up / feature explanations
├── requirements.txt      # Python dependencies
├── test_cases.json       # Example test scenarios for review
└── README.md             # This setup guide
```

---

## ✅ Final "Ready-for-Review" Checklist

Before presenting or submitting, verify these finalizing steps are done:
- [x] **Dependencies mapped**: `flask`, `torch`, `yfinance`, etc., are securely listed in `requirements.txt`.
- [x] **Dynamic Pathing Secured**: Evaluators and Trainers use `os.path` instead of hardcoded developer paths (No crash risks!).
- [x] **Database Mounted**: SQLite is implemented safely, tracking inputs/outputs natively without needing a heavy SQL server.
- [x] **Error Handling Validated**: API gracefully handles malformed data without terminal crashes.
- [x] **UI Honesty Applied**: Static frontend elements formally labeled as "(Simulated)" to satisfy strict academic review.
- [x] **Documentation Provided**: Clean README instructions compiled.

You are fully ready to demo, zip, and archive this repository!
