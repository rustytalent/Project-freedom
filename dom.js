// utils/dom.js
export let growthChart = null;
export let pnlChart = null;
export let monteCarloChart = null;
export let comparisonChart = null;

export function getElement(id) {
  return document.getElementById(id);
}

export function showNotification(message, isSuccess = false) {
  const notification = getElement('notification');
  if (!notification) {
    const n = document.createElement('div');
    n.id = 'notification';
    n.className = 'notification';
    document.body.appendChild(n);
  }
  notification.textContent = message;
  notification.className = 'notification' + (isSuccess ? ' success' : '');
  notification.style.display = 'block';
  setTimeout(() => notification.style.display = 'none', 5000);
}

export function toggleExpertModeUI(isChecked) {
  const expertSettings = getElement('expertSettings');
  expertSettings.style.display = isChecked ? 'block' : 'none';
}

export function showTab(tabName, sectionId) {
  document.querySelectorAll(`#${sectionId} .tab`).forEach(tab => tab.classList.remove('active'));
  document.querySelectorAll(`#${sectionId} .tab-content`).forEach(content => content.classList.remove('active'));
  document.querySelector(`#${sectionId} [data-tab="${tabName}"]`)?.classList.add('active');
  document.querySelector(`#${sectionId} [data-result-tab="${tabName}"]`)?.classList.add('active'); // For result tabs
  getElement(tabName)?.classList.add('active');
}

export function updateLeverageRiskUI(leverage) {
  const riskLevel = getElement('leverage-risk');
  const riskText = getElement('leverage-risk-text');
  
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
  document.querySelectorAll('input, select').forEach(el => el.classList.remove('error'));
  
  const capital = parseFloat(getElement('startingCapital').value);
  if (isNaN(capital) || capital < 1) {
    getElement('capital-error').style.display = 'block';
    getElement('startingCapital').classList.add('error');
    isValid = false;
  }
  
  const initialProfit = parseFloat(getElement('initialProfitability').value);
  if (isNaN(initialProfit) || initialProfit < 50 || initialProfit > 99) {
    getElement('initial-profit-error').style.display = 'block';
    getElement('initialProfitability').classList.add('error');
    isValid = false;
  }
  
  const finalProfit = parseFloat(getElement('finalProfitability').value);
  if (!isNaN(finalProfit) && (finalProfit < 50 || finalProfit > 90)) {
    getElement('final-profit-error').style.display = 'block';
    getElement('finalProfitability').classList.add('error');
    isValid = false;
  }
  
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
  const resultsDiv = getElement('results');
  resultsDiv.style.display = 'block';
  
  const metricsGrid = getElement('metricsGrid');
  metricsGrid.innerHTML = `
    <div class="metric">
      <div class="metric-value">$${(results.finalCapital || 0).toFixed(2)}</div>
      <div class="metric-label">Final Capital (USDT)</div>
    </div>
    <div class="metric">
      <div class="metric-value">${(results.totalGrowth || 0).toFixed(1)}%</div>
      <div class="metric-label">Total Growth</div>
    </div>
    <div class="metric">
      <div class="metric-value">${(((results.finalCapital || 0) / (startingCapital || 1)) || 0).toFixed(1)}x</div>
      <div class="metric-label">Capital Multiplier</div>
    </div>
    <div class="metric">
      <div class="metric-value">${(results.totalTrades || 0).toLocaleString()}</div>
      <div class="metric-label">Total Trades</div>
    </div>
    <div class="metric">
      <div class="metric-value">${((results.actualWinRate || 0) * 100).toFixed(1)}%</div>
      <div class="metric-label">Win Rate</div>
    </div>
    <div class="metric">
      <div class="metric-value">$${(results.totalFees || 0).toFixed(2)}</div>
      <div class="metric-label">Total Fees Paid</div>
    </div>
  `;
  
  getElement('mddValue').textContent = results.maxDrawdown.toFixed(1) + '%';
  getElement('sharpeValue').textContent = results.sharpeRatio.toFixed(2);
  getElement('sortinoValue').textContent = results.sortinoRatio.toFixed(2);
  getElement('calmarValue').textContent = results.calmarRatio.toFixed(2);
  getElement('winRateValue').textContent = (results.actualWinRate * 100).toFixed(1) + '%';
  getElement('profitFactorValue').textContent = results.profitFactor.toFixed(2);
  
  updateGrowthChart(results.dailyResults.map(r => r.capital));
  updatePnLChart(results.tradeLog.filter(t => t.isWin).length, results.tradeLog.length - results.tradeLog.filter(t => t.isWin).length);
  updateTradeLogTable(results.tradeLog);
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
      <span style="color: #00ff88">$${scenario.results.finalCapital.toFixed(2)}</span>
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
  
  updateComparisonChart(scenario1, scenario2);
}

export function displayMonteCarloStats(avgCapital, successRate, maxCapital, minCapital, profitProbability, avgDrawdown) {
  getElement('monteAvgCapital').textContent = '$' + avgCapital.toFixed(2);
  getElement('monteSuccessRate').textContent = successRate.toFixed(1) + '%';
  getElement('monteMaxCapital').textContent = '$' + maxCapital.toFixed(2);
  getElement('monteMinCapital').textContent = '$' + minCapital.toFixed(2);
  getElement('monteProbability').textContent = profitProbability.toFixed(1) + '%';
  getElement('monteAvgDrawdown').textContent = avgDrawdown.toFixed(1) + '%';
}

export function updateMonteCarloChartUI(capitalResults) {
  const ctx = getElement('monteCarloChart').getContext('2d');
  const minVal = Math.floor(Math.min(...capitalResults) / 1000) * 1000;
  const maxVal = Math.ceil(Math.max(...capitalResults) / 1000) * 1000;
  const bucketSize = Math.max(1, (maxVal - minVal) / 20);
  const buckets = Array(20).fill(0);
  
  capitalResults.forEach(capital => {
    if (capital >= minVal) {
      let bucketIndex = Math.floor((capital - minVal) / bucketSize);
      bucketIndex = Math.max(0, Math.min(buckets.length - 1, bucketIndex));
      buckets[bucketIndex]++;
    } else {
      buckets[0]++;
    }
  });
  
  const labels = [];
  for (let i = 0; i < 20; i++) {
    const start = minVal + i * bucketSize;
    const end = start + bucketSize;
    labels.push(`$${Math.round(start)}-${Math.round(end)}`);
  }
  
  if (monteCarloChart) monteCarloChart.destroy();
  monteCarloChart = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: labels,
      datasets: [{
        label: 'Final Capital Distribution',
        data: buckets,
        backgroundColor: 'rgba(0, 212, 255, 0.5)',
        borderColor: '#00d4ff',
        borderWidth: 1
      }]
    },
    options: {
      responsive: true,
      plugins: {
        legend: { labels: { color: '#b0b0ff' } }
      },
      scales: {
        x: {
          grid: { color: 'rgba(255, 255, 255, 0.05)' },
          ticks: { 
            color: '#b0b0ff',
            maxRotation: 45,
            minRotation: 45
          }
        },
        y: {
          grid: { color: 'rgba(255, 255, 255, 0.05)' },
          ticks: { color: '#b0b0ff' }
        }
      }
    }
  });
}

export function initAdvancedControlSliders() {
  document.querySelectorAll('.asset-vol').forEach(slider => {
    const valueSpan = slider.nextElementSibling;
    valueSpan.textContent = slider.value + 'x';
    slider.addEventListener('input', function() {
      valueSpan.textContent = this.value + 'x';
    });
  });
  
  document.querySelectorAll('.asset-drift').forEach(slider => {
    const valueSpan = slider.nextElementSibling;
    valueSpan.textContent = parseFloat(slider.value).toFixed(2);
    slider.addEventListener('input', function() {
      valueSpan.textContent = parseFloat(this.value).toFixed(2);
    });
  });
  
  document.querySelectorAll('.strat-weight').forEach(slider => {
    const valueDiv = slider.nextElementSibling;
    valueDiv.textContent = slider.value + '%';
    slider.addEventListener('input', function() {
      valueDiv.textContent = this.value + '%';
    });
  });
  
  const paramSliders = [
    { id: 'priceImpact', display: 'priceImpactValue' },
    { id: 'liquidityFactor', display: 'liquidityFactorValue' },
    { id: 'correlationStrength', display: 'correlationStrengthValue' },
    { id: 'newsSensitivity', display: 'newsSensitivityValue' }
  ];
  
  paramSliders.forEach(param => {
    const slider = getElement(param.id);
    const display = getElement(param.display);
    if (slider && display) {
      display.textContent = param.id === 'liquidityFactor' ? 
        parseFloat(slider.value).toFixed(1) + 'x' : 
        parseFloat(slider.value).toFixed(2);
        
      slider.addEventListener('input', function() {
        if (param.id === 'liquidityFactor') {
          display.textContent = parseFloat(this.value).toFixed(1) + 'x';
        } else {
          display.textContent = parseFloat(this.value).toFixed(2);
        }
      });
    }
  });
}

export function addNewsEventUI() {
  const newsFeed = getElement('newsFeed');
  const newItem = document.createElement('div');
  newItem.className = 'news-item';
  newItem.innerHTML = `
    <div class="news-time">Day ${Math.floor(Math.random() * 30) + 1}, ${Math.floor(Math.random() * 24)}:${Math.floor(Math.random() * 60)}</div>
    <input type="text" value="New market event" style="width:100%; background:transparent; border:1px solid #444; color:#e0e0ff; padding:3px">
  `;
  newsFeed.appendChild(newItem);
}

export function updateModelStrengthUI(strength) {
  getElement('strengthFill').style.width = `${strength}%`;
  getElement('strengthValue').textContent = `${Math.round(strength)}%`;
}

export function updateTraderProfilesUI(numTraders) {
  const traderContainer = getElement('traderProfilesContainer');
  traderContainer.innerHTML = '';
  
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
    traderContainer.appendChild(profile);
  }
}

export function updateProgressBar(containerId, fillId, percentId, percent) {
  getElement(containerId).style.display = 'block';
  getElement(fillId).style.width = percent + '%';
  getElement(percentId).textContent = percent + '%';
  if (percent === 100) {
    setTimeout(() => getElement(containerId).style.display = 'none', 500);
  }
}

// Chart.js updates (moved from original global scope)
function updateGrowthChart(data) {
  const ctx = getElement('growthChart').getContext('2d');
  const labels = data.map((_, i) => `Day ${i+1}`);
  
  if (growthChart) growthChart.destroy();
  growthChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: labels,
      datasets: [{
        label: 'Capital (USDT)',
        data: data,
        borderColor: '#00d4ff',
        backgroundColor: 'rgba(0, 212, 255, 0.1)',
        borderWidth: 2,
        pointRadius: 0,
        tension: 0.4,
        fill: true
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { labels: { color: '#b0b0ff' } }
      },
      scales: {
        x: {
          grid: { color: 'rgba(255, 255, 255, 0.05)' },
          ticks: { color: '#b0b0ff' }
        },
        y: {
          grid: { color: 'rgba(255, 255, 255, 0.05)' },
          ticks: { 
            color: '#b0b0ff',
            callback: function(value) { return '$' + value; }
          }
        }
      }
    }
  });
}

function updatePnLChart(wins, losses) {
  const pnlCtx = getElement('pnlChart').getContext('2d');
  
  if (pnlChart) pnlChart.destroy();
  pnlChart = new Chart(pnlCtx, {
    type: 'doughnut',
    data: {
      labels: ['Winning Trades', 'Losing Trades'],
      datasets: [{
        data: [wins, losses],
        backgroundColor: ['#00ff88', '#ff6b6b'],
        borderColor: ['rgba(0, 0, 0, 0.1)', 'rgba(0, 0, 0, 0.1)'],
        borderWidth: 1
      }]
    },
    options: {
      responsive: true,
      plugins: {
        legend: { labels: { color: '#b0b0ff' } }
      }
    }
  });
}

function updateTradeLogTable(tradeLog) {
  const tradeLogBody = getElement('tradeLogBody');
  tradeLogBody.innerHTML = '';
  const step = Math.max(1, Math.floor(tradeLog.length / 100)); // Sample up to 100 trades
  for (let i = 0; i < tradeLog.length; i += step) {
    const trade = tradeLog[i];
    const row = document.createElement('tr');
    row.className = trade.isWin ? 'win-trade' : 'loss-trade';
    row.innerHTML = `
      <td>${trade.day}</td>
      <td>${trade.trader}</td>
      <td>${trade.strategy || 'default'}</td>
      <td>$${trade.positionSize.toFixed(2)}</td>
      <td>${trade.isWin ? 'Win' : 'Loss'}</td>
      <td style="color: ${trade.result > 0 ? '#00ff88' : '#ff6b6b'}">${trade.result > 0 ? '+' : ''}${trade.result.toFixed(2)}</td>
      <td>$${trade.newCapital.toFixed(2)}</td>
      <td>${trade.asset}</td>
      <td>${trade.note || ''}</td>
    `;
    tradeLogBody.appendChild(row);
  }
}

function updateComparisonChart(scenario1, scenario2) {
  const ctx = getElement('comparisonChart').getContext('2d');
  const labels = scenario1.results.dailyResults.map((_, i) => `Day ${i+1}`);
  
  if (comparisonChart) comparisonChart.destroy();
  comparisonChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: labels,
      datasets: [
        {
          label: scenario1.name,
          data: scenario1.results.dailyResults.map(r => r.capital),
          borderColor: '#00d4ff',
          backgroundColor: 'rgba(0, 212, 255, 0.1)',
          borderWidth: 2,
          pointRadius: 0,
          tension: 0.4,
          fill: false
        },
        {
          label: scenario2.name,
          data: scenario2.results.dailyResults.map(r => r.capital),
          borderColor: '#9d4edd',
          backgroundColor: 'rgba(157, 78, 221, 0.1)',
          borderWidth: 2,
          pointRadius: 0,
          tension: 0.4,
          fill: false
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { labels: { color: '#b0b0ff' } }
      },
      scales: {
        x: {
          grid: { color: 'rgba(255, 255, 255, 0.05)' },
          ticks: { color: '#b0b0ff' }
        },
        y: {
          grid: { color: 'rgba(255, 255, 255, 0.05)' },
          ticks: { 
            color: '#b0b0ff',
            callback: function(value) { return '$' + value; }
          }
        }
      }
    }
  });
}
