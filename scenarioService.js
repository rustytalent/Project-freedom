// services/scenarioService.js
import { renderSavedScenariosUI } from '../utils/dom.js';

let savedScenarios = [];
export let currentResults = null;

export function saveScenario(results, config) {
  const scenario = {
    id: Date.now(),
    name: `Scenario ${savedScenarios.length + 1}`,
    config: config,
    results: results,
    timestamp: new Date()
  };
  
  savedScenarios.push(scenario);
  renderSavedScenarios();
  currentResults = results; // Set current results to the newly saved scenario
  return scenario;
}

export function renderSavedScenarios() {
  renderSavedScenariosUI(savedScenarios, (scenario) => {
    currentResults = scenario.results;
    // This will be handled by main.js in the event listener for scenario selection
  });
}

export function getSavedScenarios() {
  return savedScenarios;
}

export function setCurrentResults(results) {
  currentResults = results;
}

export function getCurrentResults() {
  return currentResults;
}

window.savedScenarios = savedScenarios;
window.currentResults = currentResults;