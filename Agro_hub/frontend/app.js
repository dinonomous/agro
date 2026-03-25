document.addEventListener('DOMContentLoaded', () => {
    console.log('Agro-Hub Dashboard Initialized');

    // Smooth entry animation for cards
    const cards = document.querySelectorAll('.glass-container');
    cards.forEach((card, index) => {
        card.style.opacity = '0';
        card.style.transform = 'translateY(20px)';
        card.style.transition = `all 0.6s cubic-bezier(0.23, 1, 0.32, 1) ${index * 0.1}s`;

        setTimeout(() => {
            card.style.opacity = '1';
            card.style.transform = 'translateY(0)';
        }, 100);
    });

    // Advisor Logic
    const historyList = document.getElementById('history-list');
    const historyForm = document.getElementById('history-form');
    const addHistoryBtn = document.getElementById('add-history');
    const advisorResult = document.getElementById('advisor-result');
    const clearHistoryBtn = document.getElementById('clear-history');
    let historyData = [];

    if (clearHistoryBtn) {
        clearHistoryBtn.addEventListener('click', () => {
            historyData = [];
            historyList.innerHTML = '';
            advisorResult.classList.add('hidden');
        });
    }

    if (addHistoryBtn) {
        addHistoryBtn.addEventListener('click', () => {
            const crop = document.getElementById('crop-type').value;
            const months = document.getElementById('duration').value;

            historyData.push({ crop, months });

            const li = document.createElement('li');
            li.className = 'history-item';
            li.innerHTML = `<span>${crop}</span> <span>${months} Months</span>`;
            historyList.appendChild(li);
        });
    }

    // Live Chart Logic
    let policyChart;
    function updateChart(suggestions) {
        const ctx = document.getElementById('policyChart');
        if (!ctx) return;

        const labels = ['Wheat', 'Rice', 'Maize', 'Soybean'];
        const dataMap = {};
        suggestions.forEach(s => dataMap[s.crop] = s.confidence * 100);
        const dataValues = labels.map(l => dataMap[l] || 0);

        if (!policyChart) {
            policyChart = new Chart(ctx, {
                type: 'doughnut',
                data: {
                    labels: labels,
                    datasets: [{
                        data: dataValues,
                        backgroundColor: ['#f59e0b', '#3b82f6', '#10b981', '#8b5cf6'],
                        borderWidth: 0,
                        hoverOffset: 10
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: { position: 'bottom', labels: { color: '#94a3b8', font: { family: 'Inter' } } },
                        tooltip: { callbacks: { label: (context) => ` ${context.label}: ${context.raw.toFixed(1)}%` } }
                    },
                    cutout: '70%'
                }
            });
        } else {
            policyChart.data.datasets[0].data = dataValues;
            policyChart.update();
        }
    }

    if (historyForm) {
        historyForm.addEventListener('submit', async (e) => {
            e.preventDefault();

            const submitBtn = historyForm.querySelector('button[type="submit"]');
            submitBtn.textContent = 'Analyzing Soil...';
            submitBtn.disabled = true;

            try {
                // Get manual soil inputs
                const manualN = document.getElementById('manual-n').value;
                const manualP = document.getElementById('manual-p').value;
                const manualK = document.getElementById('manual-k').value;
                const manualOC = document.getElementById('manual-oc').value;

                const manualSoil = {};
                if (manualN) manualSoil.n = manualN;
                if (manualP) manualSoil.p = manualP;
                if (manualK) manualSoil.k = manualK;
                if (manualOC) manualSoil.oc = manualOC;

                const requestBody = { history: historyData };
                if (Object.keys(manualSoil).length > 0) {
                    requestBody.manual_soil = manualSoil;
                }

                const response = await fetch('http://localhost:5000/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(requestBody)
                });

                const result = await response.json();

                if (result.error) throw new Error(result.error);

                // Update Suggestions List
                const suggestionsContainer = document.getElementById('suggestions-list');
                suggestionsContainer.innerHTML = ''; // Clear previous

                result.suggestions.forEach((item, index) => {
                    const card = document.createElement('div');
                    card.className = `suggestion-card ${index === 0 ? 'top-pick' : ''}`;

                    card.innerHTML = `
                        <div class="suggestion-main">
                            <div class="suggestion-header">
                                <span class="suggestion-crop">${item.crop}</span>
                                ${item.is_ai_policy_top ? '<span class="suggestion-rank">AI Policy Choice</span>' : ''}
                                ${item.is_highest_profit ? '<span class="suggestion-rank profit">Highest Profit</span>' : ''}
                                ${item.live_price !== 'N/A' ? `<span class="price-badge">₹${item.live_price}</span>` : ''}
                            </div>
                            <p class="suggestion-market">${item.market_trend}</p>
                        </div>
                        <div class="suggestion-stats">
                            <div class="suggestion-stat">
                                <span class="stat-label">Market Fit</span>
                                <span class="stat-val">${(item.confidence * 100).toFixed(1)}%</span>
                            </div>
                            <div class="suggestion-stat">
                                <span class="stat-label">Yield</span>
                                <span class="stat-val">${item.expected_yield}</span>
                            </div>
                            <div class="suggestion-stat">
                                <span class="stat-label">Soil HP</span>
                                <span class="stat-val healthy">${item.soil_health}</span>
                            </div>
                        </div>
                    `;
                    suggestionsContainer.appendChild(card);
                });

                // Update Soil Summary
                const chips = document.getElementById('soil-summary');
                chips.innerHTML = `
                    <span>Nitrogen: ${result.current_soil_summary.N}</span>
                    <span>Phos: ${result.current_soil_summary.P}</span>
                    <span>Potas: ${result.current_soil_summary.K}</span>
                    <span>Organic Carbon: ${result.current_soil_summary.OC}%</span>
                `;

                // Update Live Chart
                updateChart(result.suggestions);

                advisorResult.classList.remove('hidden');
                advisorResult.scrollIntoView({ behavior: 'smooth' });

            } catch (err) {
                alert('Error connecting to ML Advisor: ' + err.message);
            } finally {
                submitBtn.textContent = 'Get ML Recommendation';
                submitBtn.disabled = false;
            }
        });
    }

    // Load API History Logic
    const fetchHistoryBtn = document.getElementById('fetch-history-btn');
    const apiHistoryList = document.getElementById('api-history-list');

    async function loadApiHistory() {
        if (!apiHistoryList) return;
        if (fetchHistoryBtn) fetchHistoryBtn.textContent = 'Loading...';
        try {
            const response = await fetch('http://localhost:5000/history');
            const result = await response.json();
            
            if (result.error) throw new Error(result.error);
            
            apiHistoryList.innerHTML = '';
            
            if (result.history.length === 0) {
                apiHistoryList.innerHTML = '<p>No history found in database yet.</p>';
                return;
            }

            result.history.forEach(item => {
                const card = document.createElement('div');
                card.className = 'suggestion-card'; // Reuse style pattern
                
                const timeString = new Date(item.timestamp).toLocaleString();
                let topSuggestion = { crop: "Unknown", expected_yield: "N/A" };
                if (item.output_json && item.output_json.suggestions && item.output_json.suggestions.length > 0) {
                    topSuggestion = item.output_json.suggestions[0];
                }
                const inputStr = (item.input_json && item.input_json.history && item.input_json.history.length > 0) ? 
                    item.input_json.history.map(h => `${h.crop}(${h.months}m)`).join(', ') : 'Manual Soil Override';

                card.innerHTML = `
                    <div class="suggestion-main">
                        <div class="suggestion-header">
                            <span class="suggestion-crop">Request #${item.id}</span>
                            <span class="suggestion-rank profit" style="font-size:0.8rem">${timeString}</span>
                        </div>
                        <p class="suggestion-market">Input: ${inputStr}</p>
                        <p class="suggestion-market">Action: System recommended <strong style="color:var(--primary-color)">${topSuggestion.crop}</strong> (${topSuggestion.expected_yield} yield).</p>
                    </div>
                `;
                apiHistoryList.appendChild(card);
            });
        } catch (err) {
            console.error('Error loading history:', err);
            apiHistoryList.innerHTML = '<p style="color:red;font-size:0.9rem;">Error connecting to database via API.</p>';
        } finally {
            if (fetchHistoryBtn) fetchHistoryBtn.textContent = 'Refresh Activity Log';
        }
    }

    if (fetchHistoryBtn) {
        fetchHistoryBtn.addEventListener('click', loadApiHistory);
    }
    
    // Auto-load on page start
    loadApiHistory();
});
