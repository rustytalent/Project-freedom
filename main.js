// main.js
import {
  getElement, showNotification, toggleExpertModeUI, updateLeverageRiskUI,
  validateInputs, addRiskTierUI, displaySimulationResults, renderSavedScenariosUI,
  displayComparisonResults, displayMonteCarloStats,
  initAdvancedControlSliders, addNewsEventUI, updateModelStrengthUI, updateTraderProfilesUI,
  updateProgressBar,
  updateGrowthChart, updatePnLChart, updateTradeLogTable, updateComparisonChart, updateMonteCarloChart
} from './utils/dom.js';
import { EnhancedTradingModel, getConfigFromUI } from './models/tradingModel.js';
import { saveScenario, getSavedScenarios, setCurrentResults, getCurrentResults, renderSavedScenarios } from './services/scenarioService.js';

let monteCarloSimulationResults = null;

function handleCalculateProjections() {
  if (!validateInputs()) {
    showNotification('Please fix the errors in the form', false);
    return;
  }

  const config = getConfigFromUI();
  const timeframe = parseInt(getElement('timeframe').value);

  updateProgressBar('progressContainer', 'progressFill', 'progressPercent', 0);

  // Use setTimeout to allow the UI to update before the heavy calculation begins
  setTimeout(() => {
    try {
      const model = new EnhancedTradingModel(config);
      const results = model.simulate(timeframe, (percent) => {
        updateProgressBar('progressContainer', 'progressFill', 'progressPercent', percent);
      });

      setCurrentResults(results);
      displaySimulationResults(results, config.startingCapital);
      // Update charts separately
      updateGrowthChart(results.dailyResults.map(r => r.capital));
      updatePnLChart(results.winningTrades, results.totalTrades - results.winningTrades);
      updateTradeLogTable(results.tradeLog);

      showNotification('Simulation completed successfully!', true);
    } catch (error) {
      console.error("Simulation Error:", error);
      showNotification('Error during simulation: ' + error.message, false);
      updateProgressBar('progressContainer', 'progressFill', 'progressPercent', 100); // Hide progress bar on error
    }
  }, 100);
}

function handleSaveScenario() {
  const currentResults = getCurrentResults();
  if (!currentResults) {
    showNotification('Please run a simulation first', false);
    return;
  }

  const config = getConfigFromUI();
  saveScenario(currentResults, config);
  showNotification('Scenario saved successfully!', true);
}

function handleCompareScenarios() {
  const savedScenarios = getSavedScenarios();
  if (savedScenarios.length < 2) {
    showNotification('Please save at least two scenarios to compare', false);
    return;
  }

  // Find the tab button and its parent containers to activate the correct view
  const comparisonTabButton = document.querySelector('[data-tab="comparison"]');
  if (comparisonTabButton) {
      const tabGroup = comparisonTabButton.closest('.tabs');
      const card = comparisonTabButton.closest('.card');
      
      tabGroup.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
      comparisonTabButton.classList.add('active');

      card.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
      getElement('comparison').classList.add('active');
  }

  const scenario1 = savedScenarios[0];
  const scenario2 = savedScenarios[1];

  displayComparisonResults(scenario1, scenario2);
  updateComparisonChart(scenario1, scenario2);
}

function handleRunMonteCarlo() {
  if (!validateInputs()) {
    showNotification('Please fix the errors in the form', false);
    return;
  }

  const config = getConfigFromUI();
  const timeframe = parseInt(getElement('timeframe').value);
  const runs = parseInt(getElement('monteRuns').value);

  const resultsContainer = getElement('monteCarloResults');
  resultsContainer.style.display = 'none';
  updateProgressBar('monteProgressContainer', 'monteProgressFill', 'monteProgressPercent', 0);

  const allResults = [];
  const capitalResults = [];
  let completed = 0;

  const runBatch = () => {
    const batchSize = Math.min(10, runs - completed);

    for (let i = 0; i < batchSize; i++) {
      const model = new EnhancedTradingModel(config);
      const result = model.simulate(timeframe);
      allResults.push(result);
      capitalResults.push(result.finalCapital);
      completed++;
    }

    updateProgressBar('monteProgressContainer', 'monteProgressFill', 'monteProgressPercent', Math.round((completed / runs) * 100));

    if (completed < runs) {
      setTimeout(runBatch, 0); // Yield to the event loop
    } else {
      const avgCapital = capitalResults.reduce((sum, c) => sum + c, 0) / capitalResults.length;
      const minCapital = Math.min(...capitalResults);
      const maxCapital = Math.max(...capitalResults);
      const successRate = (capitalResults.filter(c => c > allResults[0].config.startingCapital).length / capitalResults.length) * 100;
      const profitProbability = successRate;
      const drawdowns = allResults.map(r => r.maxDrawdown);
      const avgDrawdown = drawdowns.reduce((sum, d) => sum + d, 0) / drawdowns.length;

      displayMonteCarloStats(avgCapital, successRate, maxCapital, minCapital, profitProbability, avgDrawdown);
      updateMonteCarloChart(capitalResults);

      monteCarloSimulationResults = allResults;
      resultsContainer.style.display = 'flex';
      setTimeout(() => getElement('monteProgressContainer').style.display = 'none', 500);
    }
  };

  runBatch();
}

function calculateModelStrength() {
    let strength = 50;
    try {
        const leverage = parseInt(getElement('leverage').value) || 25;
        const numTraders = parseInt(getElement('numTraders').value) || 2;
        const riskTolerance = getElement('riskTolerance').value;
        const initialProfit = parseFloat(getElement('initialProfitability').value) || 80;
        const finalProfit = parseFloat(getElement('finalProfitability').value) || 65;
        const maxDailyLoss = parseFloat(getElement('maxDailyLoss').value) || 5;
        const maxDrawdown = parseFloat(getElement('maxDrawdown').value) || 20;

        strength += Math.max(0, 25 - leverage);
        if (riskTolerance === 'conservative') strength += 15;
        else if (riskTolerance === 'aggressive') strength -= 10;
        strength += (initialProfit - 70) * 0.5;
        strength -= (initialProfit - finalProfit);
        strength += (numTraders - 1) * 5;
        if (maxDailyLoss < 10) strength += 5;
        if (maxDrawdown < 30) strength += 5;
        if (getElement('takeProfit').value > 0) strength += 2;
        if (getElement('trailingStop').value > 0) strength += 3;
    } catch(e) { /* ignore errors if elements not ready */ }

    return Math.max(10, Math.min(99, strength));
}


function updateOverallModelStrength() {
  const strength = calculateModelStrength();
  updateModelStrengthUI(strength);
}

document.addEventListener('DOMContentLoaded', function() {
  showNotification('Welcome to the Advanced Trading Simulator! Adjust settings and click Calculate.', true);

  updateLeverageRiskUI(parseInt(getElement('leverage').value));
  updateOverallModelStrength();
  updateTraderProfilesUI(parseInt(getElement('numTraders').value));
  initAdvancedControlSliders();

  // --- Corrected Event Listeners ---

  getElement('expertModeToggleBtn').addEventListener('click', () => {
    const checkbox = getElement('expertModeToggleCheckbox');
    checkbox.checked = !checkbox.checked;
    toggleExpertModeUI(checkbox.checked);
    updateOverallModelStrength();
  });

  // Main input tabs
  getElement('basic').parentElement.querySelector('.tabs').addEventListener('click', (e) => {
    if (e.target.matches('.tab')) {
        const tabGroup = e.target.parentElement;
        const card = tabGroup.parentElement;
        const tabName = e.target.dataset.tab;

        tabGroup.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
        e.target.classList.add('active');

        card.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
        getElement(tabName)?.classList.add('active');
    }
  });

  // Results tabs
  getElement('results').querySelector('.tabs').addEventListener('click', (e) => {
      if(e.target.matches('.tab')) {
        const tabGroup = e.target.parentElement;
        const resultsContainer = tabGroup.parentElement;
        const tabName = e.target.dataset.resultTab;

        tabGroup.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
        e.target.classList.add('active');

        resultsContainer.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
        getElement(tabName)?.classList.add('active');
      }
  });

  getElement('leverage').addEventListener('change', () => {
    updateLeverageRiskUI(parseInt(getElement('leverage').value));
    updateOverallModelStrength();
  });

  const strengthInputs = ['riskTolerance', 'numTraders', 'initialProfitability', 'finalProfitability', 'maxDailyLoss', 'maxDrawdown', 'takeProfit', 'trailingStop'];
  strengthInputs.forEach(id => getElement(id)?.addEventListener('change', updateOverallModelStrength));

  getElement('numTraders').addEventListener('change', () => {
    const count = parseInt(getElement('numTraders').value);
    updateTraderProfilesUI(count);
  });
  
  getElement('calculateProjectionsBtn').addEventListener('click', handleCalculateProjections);
  getElement('addRiskTierBtn').addEventListener('click', addRiskTierUI);
  getElement('addNewsEventBtn').addEventListener('click', addNewsEventUI);
  getElement('saveScenarioBtn').addEventListener('click', handleSaveScenario);
  getElement('compareScenariosBtn').addEventListener('click', handleCompareScenarios);
  getElement('runMonteCarloBtn').addEventListener('click', handleRunMonteCarlo);
  getElement('uploadCsvBtn').addEventListener('click', () => getElement('csvUpload').click());

  getElement('csvUpload').addEventListener('change', async e => {
    const file = e.target.files[0];
    if (!file) return;

    try {
      showNotification('CSV processing is simulated in this demo', true);
      const config = getConfigFromUI();
      const model = new EnhancedTradingModel(config);
      const timeframe = parseInt(getElement('timeframe').value);
      const results = model.simulate(timeframe);
      setCurrentResults(results);
      displaySimulationResults(results, config.startingCapital);
      updateGrowthChart(results.dailyResults.map(r => r.capital));
      updatePnLChart(results.winningTrades, results.totalTrades - results.winningTrades);
      updateTradeLogTable(results.tradeLog);
      showNotification('CSV back-test completed successfully!', true);
    } catch (error) {
      showNotification('Error processing CSV: ' + error.message, false);
    }
  });

  renderSavedScenarios();
});
