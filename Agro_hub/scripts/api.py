import os
import sys
import torch
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from stable_baselines3 import PPO

# Add the parent directory to sys.path to allow importing 'env' and 'scripts'
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from env.agro_env import AgroEnv
import copy
import yfinance as yf
from datetime import datetime, timedelta

# Import the database helper
from scripts.db import save_request, get_recent_requests

app = Flask(__name__)
CORS(app)
import copy
import yfinance as yf
from datetime import datetime, timedelta

app = Flask(__name__)
CORS(app)

# Load the trained model
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'ppo_agro_final')
model = None

def get_model():
    global model
    if model is None:
        model = PPO.load(MODEL_PATH)
    return model

# Tickers for commodities
TICKERS = {
    "Wheat": "ZW=F",
    "Rice": "ZR=F",
    "Maize": "ZC=F",
    "Soybean": "ZS=F",
    "USDINR": "USDINR=X" # Live exchange rate
}

def get_live_market_data():
    market_data = {}
    try:
        data = yf.download(list(TICKERS.values()), period="5d", interval="1d", group_by='ticker', progress=False)
        
        # Get live USD to INR rate
        try:
            inr_rate = data["USDINR=X"]['Close'].iloc[-1]
        except:
            inr_rate = 83.50 # Fallback rate
            
        for crop, ticker in TICKERS.items():
            if crop == "USDINR": continue
            try:
                ticker_data = data[ticker]
                current_price_usd = ticker_data['Close'].iloc[-1]
                prev_price_usd = ticker_data['Close'].iloc[-2]
                
                # Convert to INR
                current_price = current_price_usd * inr_rate
                prev_price = prev_price_usd * inr_rate
                
                change_pct = ((current_price - prev_price) / prev_price) * 100
                
                trend = "Bullish (Rising)" if change_pct > 0 else "Bearish (Falling)"
                if abs(change_pct) < 0.5: trend = "Neutral (Stable)"
                
                market_data[crop] = {
                    "price": f"{current_price:.2f}",
                    "change": f"{change_pct:+.2f}%",
                    "trend": trend,
                    "insight": f"Price is {trend} at ₹{current_price:,.2f} per unit."
                }
            except:
                market_data[crop] = {"price": "N/A", "change": "0.00%", "trend": "Unknown", "insight": "Market data currently unavailable."}
    except Exception as e:
        print(f"Error fetching market data: {e}")
        return None
    return market_data

@app.route('/predict', methods=['POST'])
def predict():
    """
    Core ML endpoint.
    Retrieves user history / manual soil overrides, simulates the soil state,
    and queries the trained PPO RL model and external market data to provide
    the best crop recommendations.
    """
    try:
        # 0. Basic Validation
        if not request.is_json:
            return jsonify({"error": "Request must be JSON"}), 400
            
        data = request.json
        if not isinstance(data, dict):
            return jsonify({"error": "Invalid JSON mapping format."}), 400
            
        history = data.get('history', [])
        manual_soil = data.get('manual_soil', {})
        
        env = AgroEnv()
        obs, _ = env.reset()
        
        # 1. State Initializer (Manual Overrides first, then History)
        # This overrides the default or simulated state to allow the user
        # to test real current-day field conditions.
        if manual_soil:
            if 'n' in manual_soil: obs[0] = float(manual_soil['n'])
            if 'p' in manual_soil: obs[1] = float(manual_soil['p'])
            if 'k' in manual_soil: obs[2] = float(manual_soil['k'])
            if 'oc' in manual_soil: obs[5] = float(manual_soil['oc'])
        
        crop_map = {"Wheat": 0, "Rice": 1, "Maize": 2, "Soybean": 3}
        
        # 2. Simulate History 
        # Walks through the environment 'step' function to degrade the
        # soil matching the user's past cultivation choices.
        for entry in history:
            crop_name = str(entry.get('crop', ''))
            if crop_name not in crop_map:
                continue
            months = int(entry.get('months', 4))
            cycles = max(1, months // 3) 
            action = crop_map.get(crop_name, 0)
            for _ in range(cycles):
                obs, _, _, _, _ = env.step(action)
        
        # 3. Analyze current soil state via RL model
        # Loads the PPO model and asks for a probability distribution.
        try:
            algo = get_model()
            obs_tensor = torch.as_tensor(obs).unsqueeze(0)
            with torch.no_grad():
                distribution = algo.policy.get_distribution(obs_tensor)
                probs = torch.softmax(distribution.distribution.logits, dim=1).numpy()[0]
        except Exception as e:
            return jsonify({"error": f"Model inference error: {str(e)}"}), 500
        
        # Fetch live market data (if API rate limited, it returns fallback data)
        live_data = get_live_market_data()
        
        # 4. Generate suggestions for supported crops
        suggestions = []
        base_insights = {
            "Wheat": "High domestic stability.",
            "Rice": "Global supply constraint.",
            "Maize": "Bio-fuel demand rising.",
            "Soybean": "Protein demand at record high."
        }

        # Validate the environment step sizes
        for i in range(4): # 0 to 3: Wheat, Rice, Maize, Soybean
            crop_name = env.crop_names[i]
            test_env = copy.deepcopy(env)
            test_env.state = obs.copy()
            _, reward, _, _, info = test_env.step(i)
            
            market_info = live_data[crop_name] if live_data and crop_name in live_data else {"trend": "Stable", "change": "0%", "insight": base_insights[crop_name]}
            
            suggestions.append({
                "crop": crop_name,
                "confidence": float(probs[i]),
                "expected_yield": f"{info['yield']:.2f}x",
                "soil_health": f"{info['soil_health']:.2f}",
                "market_trend": f"{market_info['trend']} ({market_info['change']}). {market_info['insight']}",
                "live_price": market_info.get("price", "N/A"),
                "score": float(reward)
            })
        
        # 5. Hybrid Ranking (Rank by Confidence/Market Fit)
        suggestions = sorted(suggestions, key=lambda x: x['confidence'], reverse=True)
        
        # Identify which one has the best raw yield/profit score (Greedy benchmark)
        max_score = -1e9
        max_score_idx = 0
        for idx, s in enumerate(suggestions):
            if s['score'] > max_score:
                max_score = s['score']
                max_score_idx = idx
        
        for idx, s in enumerate(suggestions):
            s['is_highest_profit'] = (idx == max_score_idx)
            s['is_ai_policy_top'] = (idx == 0)

        response_payload = {
            "suggestions": suggestions,
            "current_soil_summary": {
                "N": f"{obs[0]:.1f}", 
                "P": f"{obs[1]:.1f}", 
                "K": f"{obs[2]:.1f}",
                "pH": f"{obs[3]:.2f}",
                "OC": f"{obs[5]:.2f}"
            }
        }
        
        # Save request and response to SQLite database asynchronously 
        #/ in the background (preventing blockers for the view)
        save_request(data, response_payload)
        
        return jsonify(response_payload)

    except Exception as e:
        import traceback
        traceback.print_exc()
        # Return a cleanly structured error standard
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

@app.route('/history', methods=['GET'])
def get_history():
    """
    Returns the last 10 requests from the SQLite database.
    Used to demonstrate application state and log tracking.
    """
    try:
        data = get_recent_requests(10)
        return jsonify({"history": data}), 200
    except Exception as e:
        return jsonify({"error": f"Failed to fetch history: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(port=5000)
