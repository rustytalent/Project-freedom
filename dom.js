// utils/dom.js
let growthChart = null;
let pnlChart = null;
let monteCarloChart = null;
let comparisonChart = null;

export function getElement(id) {
  return document.getElementById(id);
}

export function showNotification(message, isSuccess = false) {
  const notification = getElement('notification');
  if (!notification) return;
  notification.textContent = message;
  notification.className = 'notification' + (isSuccess ? ' success' : '');
  notification.style.display = 'block';
  setTimeout(() => {
      if (notification) notification.style.display = 'none';
  }, 5000);
}

export function toggleExpertModeUI(isChecked) {
  getElement('expertSettings').style.display = isChecked ? 'block' : 'none';
}

export function updateLeverageRiskUI(leverage) {
  const riskLevel = getElement('leverage-risk');
  const riskText = getElement('leverage-risk-text');
  if (!riskLevel || !riskText) return;

  if (leverage <= 10) {
    riskLevel.className = 'risk-level';
    riskText.textContent = 'Low Risk';
  } else if (leverage <= 25) {
    riskLevel.className = 'risk-level medium';
    riskText.textContent = 'Moderate Risk';
  } else {
    riskLevel.className = 'risk-level high';
    riskText.textContent = 'High Risk';
  }
}

export function validateInputs() {
  let isValid = true;
  document.querySelectorAll('.error-message').forEach(el => (el.style.display = 'none'));
  document.querySelectorAll('input.error, select.error').forEach(el => el.classList.remove('error'));

  const validateField = (id, errorId, condition) => {
    const field = getElement(id);
    const errorField = getElement(errorId);
    if (condition(field.value)) {
      errorField.style.display = 'block';
      field.classList.add('error');
      isValid = false;
    }
  };

  validateField('startingCapital', 'capital-error', val => isNaN(parseFloat(val)) || parseFloat(val) < 1);
  validateField('initialProfitability', 'initial-profit-error', val => isNaN(parseFloat(val)) || val < 50 || val > 99);
  validateField('finalProfitability', 'final-profit-error', val => isNaN(parseFloat(val)) || val < 50 || val > 90);
  
  return isValid;
}

export function addRiskTierUI() {
  const container = getElement('riskTiersContainer');
  const newRow = document.createElement('div');
  newRow.className = 'risk-tier-row';
  newRow.innerHTML = `
    <input type="number" class="risk-tier-threshold" placeholder="Threshold (USDT)" value="0" min="0">
    <input type="number" class="risk-tier-percent" placeholder="Risk %" value="0.1" min="0.01" max="1" step="0.01">
  `;
  container.appendChild(newRow);
}

export function displaySimulationResults(results, startingCapital) {
  getElement('results').style.display = 'block';

  const metricsGrid = getElement('metricsGrid');
  metricsGrid.innerHTML = `
    <div class="metric"><div class="metric-value">$${(results.finalCapital || 0).toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2})}</div><div class="metric-label">Final Capital (USDT)</div></div>
    <div class="metric"><div class="metric-value">${(results.totalGrowth || 0).toFixed(1)}%</div><div class="metric-label">Total Growth</div></div>
    <div class="metric"><div class="metric-value">${(((results.finalCapital || 0) / (startingCapital || 1)) || 0).toFixed(1)}x</div><div class="metric-label">Capital Multiplier</div></div>
    <div class="metric"><div class="metric-value">${(results.totalTrades || 0).toLocaleString()}</div><div class="metric-label">Total Trades</div></div>
    <div class="metric"><div class="metric-value">${((results.actualWinRate || 0) * 100).toFixed(1)}%</div><div class="metric-label">Win Rate</div></div>
    <div class="metric"><div class="metric-value">$${(results.totalFees || 0).toFixed(2)}</div><div class="metric-label">Total Fees Paid</div></div>
  `;

  getElement('mddValue').textContent = results.maxDrawdown.toFixed(1) + '%';
  getElement('sharpeValue').textContent = results.sharpeRatio.toFixed(2);
  getElement('sortinoValue').textContent = results.sortinoRatio.toFixed(2);
  getElement('calmarValue').textContent = results.calmarRatio.toFixed(2);
  getElement('winRateValue').textContent = (results.actualWinRate * 100).toFixed(1) + '%';
  getElement('profitFactorValue').textContent = results.profitFactor.toFixed(2);
}

export function renderSavedScenariosUI(savedScenarios, onSelectScenario) {
  const container = getElement('savedScenarios');
  container.innerHTML = '';

  savedScenarios.forEach((scenario) => {
    const tag = document.createElement('div');
    tag.className = 'scenario-tag';
    tag.innerHTML = `
      <i class="fas fa-folder"></i>
      ${scenario.name}
      <span style="color: #00ff88">$${scenario.results.finalCapital.toFixed(0)}</span>
    `;
    tag.addEventListener('click', () => {
      document.querySelectorAll('.scenario-tag').forEach(t => t.classList.remove('active'));
      tag.classList.add('active');
      onSelectScenario(scenario);
    });
    container.appendChild(tag);
  });
}

export function displayComparisonResults(scenario1, scenario2) {
  getElement('compFinalCapital1').textContent = '$' + scenario1.results.finalCapital.toFixed(2);
  getElement('compFinalCapital2').textContent = '$' + scenario2.results.finalCapital.toFixed(2);
  getElement('compGrowth1').textContent = scenario1.results.totalGrowth.toFixed(1) + '%';
  getElement('compGrowth2').textContent = scenario2.results.totalGrowth.toFixed(1) + '%';
  getElement('compMDD1').textContent = scenario1.results.maxDrawdown.toFixed(1) + '%';
  getElement('compMDD2').textContent = scenario2.results.maxDrawdown.toFixed(1) + '%';
  getElement('compSharpe1').textContent = scenario1.results.sharpeRatio.toFixed(2);
  getElement('compSharpe2').textContent = scenario2.results.sharpeRatio.toFixed(2);
  getElement('compSortino1').textContent = scenario1.results.sortinoRatio.toFixed(2);
  getElement('compSortino2').textContent = scenario2.results.sortinoRatio.toFixed(2);
  getElement('compCalmar1').textContent = scenario1.results.calmarRatio.toFixed(2);
  getElement('compCalmar2').textContent = scenario2.results.calmarRatio.toFixed(2);
}

export function displayMonteCarloStats(avgCapital, successRate, maxCapital, minCapital, profitProbability, avgDrawdown) {
  getElement('monteAvgCapital').textContent = '$' + avgCapital.toFixed(2);
  getElement('monteSuccessRate').textContent = successRate.toFixed(1) + '%';
  getElement('monteMaxCapital').textContent = '$' + maxCapital.toFixed(2);
  getElement('monteMinCapital').textContent = '$' + minCapital.toFixed(2);
  getElement('monteProbability').textContent = profitProbability.toFixed(1) + '%';
  getElement('monteAvgDrawdown').textContent = avgDrawdown.toFixed(1) + '%';
}

export function initAdvancedControlSliders() {
  const setupSlider = (selector, valueFormatter) => {
    document.querySelectorAll(selector).forEach(slider => {
      const display = slider.nextElementSibling;
      display.textContent = valueFormatter(slider.value);
      slider.addEventListener('input', function() {
        display.textContent = valueFormatter(this.value);
      });
    });
  };

  setupSlider('.asset-vol', val => val + 'x');
  setupSlider('.asset-drift', val => parseFloat(val).toFixed(2));
  setupSlider('.strat-weight', val => val + '%');

  const paramSliders = [
    { id: 'priceImpact', displayId: 'priceImpactValue', formatter: val => parseFloat(val).toFixed(2) },
    { id: 'liquidityFactor', displayId: 'liquidityFactorValue', formatter: val => parseFloat(val).toFixed(1) + 'x' },
    { id: 'correlationStrength', displayId: 'correlationStrengthValue', formatter: val => parseFloat(val).toFixed(1) },
    { id: 'newsSensitivity', displayId: 'newsSensitivityValue', formatter: val => parseFloat(val).toFixed(1) },
  ];
  paramSliders.forEach(({id, displayId, formatter}) => {
      const slider = getElement(id);
      const display = getElement(displayId);
      if (slider && display) {
          display.textContent = formatter(slider.value);
          slider.addEventListener('input', function() {
              display.textContent = formatter(this.value);
          });
      }
  });
}

export function addNewsEventUI() {
  const newsFeed = getElement('newsFeed');
  const newItem = document.createElement('div');
  newItem.className = 'news-item';
  newItem.innerHTML = `
    <div class="news-time">Day ${Math.floor(Math.random() * 30) + 1}, ${String(Math.floor(Math.random() * 24)).padStart(2,'0')}:${String(Math.floor(Math.random() * 60)).padStart(2,'0')}</div>
    <input type="text" value="New market event" style="width:100%; background:transparent; border:1px solid #444; color:#e0e0ff; padding:3px; border-radius: 3px;">
  `;
  newsFeed.appendChild(newItem);
  newsFeed.scrollTop = newsFeed.scrollHeight;
}

export function updateModelStrengthUI(strength) {
  getElement('strengthFill').style.width = `${strength}%`;
  getElement('strengthValue').textContent = `${Math.round(strength)}%`;
}

export function updateTraderProfilesUI(numTraders) {
  const container = getElement('traderProfilesContainer');
  container.innerHTML = '';
  for (let i = 1; i <= numTraders; i++) {
    const profile = document.createElement('div');
    profile.className = 'trader-profile';
    profile.innerHTML = `
      <h4><i class="fas fa-user"></i> Trader ${i}</h4>
      <div class="form-group">
        <label for="trader${i}Efficiency">Base Efficiency</label>
        <input type="number" id="trader${i}Efficiency" class="trader-efficiency" value="1.0" min="0.5" max="1.5" step="0.1">
      </div>
    `;
    container.appendChild(profile);
  }
}

export function updateProgressBar(containerId, fillId, percentId, percent) {
  const container = getElement(containerId);
  if (!container) return;
  container.style.display = 'block';
  getElement(fillId).style.width = percent + '%';
  getElement(percentId).textContent = percent + '%';
  if (percent >= 100) {
    setTimeout(() => container.style.display = 'none', 1000);
  }
}

// --- Charting Functions ---

export function updateGrowthChart(data) {
  const ctx = getElement('growthChart').getContext('2d');
  const labels = data.map((_, i) => `Day ${i + 1}`);
  if (growthChart) growthChart.destroy();
  growthChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: labels,
      datasets: [{
        label: 'Capital (USDT)', data: data, borderColor: '#00d4ff', backgroundColor: 'rgba(0, 212, 255, 0.1)',
        borderWidth: 2, pointRadius: 0, tension: 0.4, fill: true
      }]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { labels: { color: '#b0b0ff' } } },
      scales: {
        x: { grid: { color: 'rgba(255, 255, 255, 0.05)' }, ticks: { color: '#b0b0ff' } },
        y: { grid: { color: 'rgba(255, 255, 255, 0.05)' }, ticks: { color: '#b0b0ff', callback: value => '$' + value.toLocaleString() } }
      }
    }
  });
}

export function updatePnLChart(wins, losses) {
  const ctx = getElement('pnlChart').getContext('2d');
  if (pnlChart) pnlChart.destroy();
  pnlChart = new Chart(ctx, {
    type: 'doughnut',
    data: {
      labels: ['Winning Trades', 'Losing Trades'],
      datasets: [{
        data: [wins, losses],
        backgroundColor: ['#00ff88', '#ff6b6b'],
        borderColor: ['#1a1a3e', '#1a1a3e'], borderWidth: 3
      }]
    },
    options: { responsive: true, plugins: { legend: { position: 'top', labels: { color: '#b0b0ff' } } } }
  });
}

export function updateTradeLogTable(tradeLog) {
  const tradeLogBody = getElement('tradeLogBody');
  tradeLogBody.innerHTML = '';
  const step = Math.max(1, Math.floor(tradeLog.length / 100)); // Sample up to 100 trades
  for (let i = 0; i < tradeLog.length; i += step) {
    const trade = tradeLog[i];
    const row = document.createElement('tr');
    row.className = trade.isWin ? 'win-trade' : 'loss-trade';
    row.innerHTML = `
      <td>${trade.day}</td> <td>${trade.trader}</td> <td>${trade.strategy || 'default'}</td>
      <td>$${trade.positionSize.toFixed(2)}</td> <td>${trade.isWin ? 'Win' : 'Loss'}</td>
      <td style="color: ${trade.result > 0 ? '#00ff88' : '#ff6b6b'}">${trade.result > 0 ? '+' : ''}${trade.result.toFixed(2)}</td>
      <td>$${trade.newCapital.toFixed(2)}</td> <td>${trade.asset}</td> <td>${trade.note || ''}</td>
    `;
    tradeLogBody.appendChild(row);
  }
}

export function updateComparisonChart(scenario1, scenario2) {
  const ctx = getElement('comparisonChart').getContext('2d');
  const labels = scenario1.results.dailyResults.map((_, i) => `Day ${i + 1}`);
  if (comparisonChart) comparisonChart.destroy();
  comparisonChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: labels,
      datasets: [
        { label: scenario1.name, data: scenario1.results.dailyResults.map(r => r.capital), borderColor: '#00d4ff', tension: 0.4, pointRadius: 0, fill: false },
        { label: scenario2.name, data: scenario2.results.dailyResults.map(r => r.capital), borderColor: '#9d4edd', tension: 0.4, pointRadius: 0, fill: false }
      ]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { labels: { color: '#b0b0ff' } } },
      scales: {
        x: { grid: { color: 'rgba(255, 255, 255, 0.05)' }, ticks: { color: '#b0b0ff' } },
        y: { grid: { color: 'rgba(255, 255, 255, 0.05)' }, ticks: { color: '#b0b0ff', callback: value => '$' + value.toLocaleString() } }
      }
    }
  });
}

export function updateMonteCarloChart(capitalResults) {
  const ctx = getElement('monteCarloChart').getContext('2d');
  const minVal = Math.min(...capitalResults);
  const maxVal = Math.max(...capitalResults);
  const numBuckets = 20;
  const bucketSize = (maxVal - minVal) / numBuckets || 1;
  const buckets = Array(numBuckets).fill(0);
  
  capitalResults.forEach(capital => {
      let bucketIndex = Math.floor((capital - minVal) / bucketSize);
      bucketIndex = Math.min(numBuckets - 1, Math.max(0, bucketIndex));
      buckets[bucketIndex]++;
  });
  
  const labels = Array.from({length: numBuckets}, (_, i) => `$${(minVal + i * bucketSize).toFixed(0)}`);

  if (monteCarloChart) monteCarloChart.destroy();
  monteCarloChart = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: labels,
      datasets: [{
        label: 'Final Capital Distribution', data: buckets,
        backgroundColor: 'rgba(0, 212, 255, 0.5)', borderColor: '#00d4ff', borderWidth: 1
      }]
    },
    options: {
      responsive: true,
      plugins: { legend: { display: false } },
      scales: {
        x: { grid: { color: 'rgba(255, 255, 255, 0.05)' }, ticks: { color: '#b0b0ff', maxRotation: 45, minRotation: 45 } },
        y: { grid: { color: 'rgba(255, 255, 255, 0.05)' }, ticks: { color: '#b0b0ff' }, title: { display: true, text: 'Frequency', color: '#b0b0ff' } }
      }
    }
  });
}
