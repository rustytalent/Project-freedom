// utils/charting.js
import { getElement } from './dom.js';

let growthChartInstance = null;
let pnlChartInstance = null;
let monteCarloChartInstance = null;
let comparisonChartInstance = null;

export function updateGrowthChart(data) {
  const ctx = getElement('growthChart').getContext('2d');
  const labels = data.map((_, i) => `Day ${i+1}`);
  
  if (growthChartInstance) growthChartInstance.destroy();
  growthChartInstance = new Chart(ctx, {
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

export function updatePnLChart(wins, losses) {
  const pnlCtx = getElement('pnlChart').getContext('2d');
  
  if (pnlChartInstance) pnlChartInstance.destroy();
  pnlChartInstance = new Chart(pnlCtx, {
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

export function updateMonteCarloChart(capitalResults) {
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
  
  if (monteCarloChartInstance) monteCarloChartInstance.destroy();
  monteCarloChartInstance = new Chart(ctx, {
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

export function updateComparisonChart(scenario1, scenario2) {
  const ctx = getElement('comparisonChart').getContext('2d');
  const labels = scenario1.results.dailyResults.map((_, i) => `Day ${i+1}`);
  
  if (comparisonChartInstance) comparisonChartInstance.destroy();
  comparisonChartInstance = new Chart(ctx, {
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
