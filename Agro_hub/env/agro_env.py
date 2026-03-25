import gymnasium as gym
from gymnasium import spaces
import numpy as np

class AgroEnv(gym.Env):
    """
    Custom Environment for Soil-Aware Crop Rotation Optimization.
    """
    metadata = {"render_modes": ["human"]}

    def __init__(self):
        super(AgroEnv, self).__init__()

        # Action space: 0: Wheat, 1: Rice, 2: Maize, 3: Soybean, 4: Pulses
        self.action_space = spaces.Discrete(5)
        self.crop_names = ["Wheat", "Rice", "Maize", "Soybean", "Pulses"]

        # Observation space:
        # N, P, K (0-500 kg/ha)
        # pH (0-14, though usually 4.5-9)
        # Moisture (0-100%)
        # Organic Carbon (0-5%)
        # Temperature (0-50 C)
        # Rainfall (0-1000 mm)
        low = np.array([0, 0, 0, 0, 0, 0, -10, 0], dtype=np.float32)
        high = np.array([500, 500, 500, 14, 100, 5, 50, 1000], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # Agronomic constants (Simplified synthetic model)
        # uptake: [N, P, K, Water]
        self.crop_uptake = {
            0: [100, 30, 80, 400],    # Wheat
            1: [150, 40, 100, 1200],  # Rice
            2: [120, 35, 90, 600],    # Maize
            3: [-20, 25, 60, 450],    # Soybean (N-Fixation)
            4: [-40, 20, 40, 300]     # Pulses (High N-Fixation)
        }
        
        # Ideal soil conditions [N, P, K, pH_min, pH_max, Moisture_min]
        self.crop_requirements = {
            0: [80, 20, 60, 6.0, 7.5, 30],
            1: [120, 30, 80, 5.5, 7.0, 80],
            2: [100, 25, 70, 5.8, 7.2, 40],
            3: [20, 20, 50, 6.0, 7.0, 35],
            4: [10, 15, 40, 6.0, 7.5, 25]
        }

        self.state = None
        self.current_step = 0
        self.max_steps = 40  # 10 years (4 seasons/year)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initial healthy soil state
        n = np.random.uniform(150, 250)
        p = np.random.uniform(40, 60)
        k = np.random.uniform(150, 200)
        ph = np.random.uniform(6.0, 7.0)
        moisture = np.random.uniform(40, 60)
        oc = np.random.uniform(1.5, 2.5)
        temp = self._get_seasonal_temp()
        rain = self._get_seasonal_rain()

        self.state = np.array([n, p, k, ph, moisture, oc, temp, rain], dtype=np.float32)
        self.current_step = 0
        
        return self.state, {}

    def _get_seasonal_temp(self):
        # Sine wave simulation for seasons
        return 25 + 10 * np.sin(2 * np.pi * self.current_step / 4)

    def _get_seasonal_rain(self):
        # Stochastic rainfall with seasonal bias
        base_rain = 200 + 300 * np.maximum(0, np.sin(2 * np.pi * self.current_step / 4))
        return np.random.normal(base_rain, 50)

    def step(self, action):
        n, p, k, ph, moisture, oc, temp, rain = self.state
        uptake = self.crop_uptake[action]
        req = self.crop_requirements[action]

        # 1. Calculate Yield Performance (Liebig's Law of the Minimum)
        # Yield is limited by the most scarce resource
        n_factor = np.clip(n / req[0], 0, 1.2) if req[0] > 0 else 1.2
        p_factor = np.clip(p / req[1], 0, 1.2) if req[1] > 0 else 1.2
        k_factor = np.clip(k / req[2], 0, 1.2) if req[2] > 0 else 1.2
        
        # pH factor: Gaussian-like response centered on ideal pH range
        ph_mid = (req[3] + req[4]) / 2
        ph_width = (req[4] - req[3])
        ph_factor = np.exp(-((ph - ph_mid)**2) / (2 * (ph_width**2)))
        
        # Water factor: excess water (Rice) or drought stress
        current_water = moisture + rain/10
        water_factor = np.clip(current_water / req[5], 0, 1.5) if req[5] > 0 else 1.0
        if action != 1 and current_water > 150: # Excess water penalty for non-rice
            water_factor *= 0.7

        # Final Yield is the product of factors (Multi-objective optimization target)
        yield_score = min(n_factor, p_factor, k_factor) * ph_factor * water_factor
        yield_score = np.clip(yield_score, 0, 1.5)

        # 2. Update Soil State
        # Nutrient depletion proportional to yield
        n = np.clip(n - uptake[0] * yield_score + np.random.normal(10, 2), 0, 500)
        p = np.clip(p - uptake[1] * yield_score + np.random.normal(2, 0.5), 0, 500)
        k = np.clip(k - uptake[2] * yield_score + np.random.normal(5, 1), 0, 500)
        
        # pH dynamics: Nitrogen fertilizers (non-legumes) acidify soil; Legumes have a buffering effect
        if action in [3, 4]: # Soybean, Pulses
            ph_change = np.random.normal(0.02, 0.01)
        else:
            ph_change = np.random.normal(-0.03, 0.01)
        ph = np.clip(ph + ph_change, 4.0, 9.0)
        
        # Organic carbon logic: Improved by legumes and residue
        oc_change = 0.08 * yield_score if action in [3, 4] else -0.04
        oc = np.clip(oc + oc_change, 0.1, 5.0)

        # 3. Calculate Reward (The core of the DRL objective)
        # Components: Yield, Sustainability Index, Degradation Penalty
        soil_health = (n/300 + p/60 + k/200 + oc/3.0) / 4.0
        sustainability_reward = soil_health * 2.0
        
        # Penalty for extreme conditions (Soil Degradation)
        degradation_penalty = 0
        if n < 30 or p < 5 or k < 30 or oc < 0.4 or not (5.0 <= ph <= 8.5):
            degradation_penalty = 10.0
            
        reward = (yield_score * 15.0) + sustainability_reward - degradation_penalty
        
        # 4. Seasonal Transitions
        self.current_step += 1
        next_temp = self._get_seasonal_temp()
        next_rain = self._get_seasonal_rain()
        
        # Moisture depends on rain, evaporation (temp), and previous moisture
        evaporation = (temp / 20) * 5
        next_moisture = np.clip(moisture * 0.4 + rain/15 - evaporation - uptake[3]/800, 0, 100)
        
        self.state = np.array([n, p, k, ph, next_moisture, oc, next_temp, next_rain], dtype=np.float32)
        
        terminated = self.current_step >= self.max_steps
        truncated = False
        
        return self.state, reward, terminated, truncated, {"yield": yield_score, "soil_health": soil_health}

    def render(self):
        print(f"Step: {self.current_step}, State: {self.state}")
