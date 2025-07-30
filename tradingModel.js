// models/tradingModel.js
import { getElement } from '../utils/dom.js';

export class EnhancedTradingModel {
  constructor(config) {
    this.config = {
      startingCapital: 1000,
      leverage: 25,
      tradingHours: 8,
      numTraders: 2,
      initialProfitability: 0.80,
      finalProfitability: 0.65,
      marketCondition: 'normal',
      compoundFrequency: 'daily',
      riskTolerance: 'moderate',
      baseVolatility: 1.0,
      cyclicalPeriod: 14,
      ...config
    };
    
    this.riskTiers = this.getRiskTiersFromUI();
    this.traderProfiles = [];
    this.strategies = [
      { name: 'scalp', baseEdge: 0.75, volMult: 1.2, decay: 1.5 },
      { name: 'swing', baseEdge: 0.65, volMult: 1.0, decay: 1.0 },
      { name: 'breakout', baseEdge: 0.70, volMult: 1.4, decay: 1.2 }
    ];
    
    for (let i = 0; i < this.config.numTraders; i++) {
      this.traderProfiles.push({
        id: `TRADER_${i + 1}`,
        baseEfficiency: this.getTraderEfficiency(i + 1),
        fatigue: 0,
        streak: 0
      });
    }
    
    this.stopConfig = {
      maxDailyLoss: parseFloat(getElement('maxDailyLoss').value || 5) / 100,
      maxDrawdown: parseFloat(getElement('maxDrawdown').value || 20) / 100,
      takeProfit: parseFloat(getElement('takeProfit').value || 10) / 100,
      trailingStop: parseFloat(getElement('trailingStop').value || 5) / 100
    };
    
    this.assets = this.getAssetsFromUI();
    this.strategyWeights = this.getStrategyWeights();
    this.economicEvents = this.getEconomicEvents();
    
    this.priceImpact = parseFloat(getElement('priceImpact').value || 0.05);
    this.liquidityFactor = parseFloat(getElement('liquidityFactor').value || 1.0);
    this.correlationStrength = parseFloat(getElement('correlationStrength').value || 0.7);
    this.newsSensitivity = parseFloat(getElement('newsSensitivity').value || 0.5);
    
    this.correlationMatrix = this.getCorrelationMatrix();
    this.newsEvents = this.getNewsEvents();
    
    this.consecutiveLosses = 0;
    this.consecutiveWins = 0;
    this.tradeLog = [];
    this.behaviourPhase = 0;
    this.phaseMemory = [];
    this.fatigueState = { mental: 0, lastRest: 0 };
    this.peakCapital = this.config.startingCapital;
  }
  
  getAssetsFromUI() {
    const assets = [];
    document.querySelectorAll('.asset-row').forEach(row => {
      const name = row.querySelector('.asset-name').textContent.trim();
      const vol = parseFloat(row.querySelector('.asset-vol').value);
      const drift = parseFloat(row.querySelector('.asset-drift').value);
      assets.push({ name, vol, drift, correlation: 0.7 });
    });
    return assets.length ? assets : [
      { name: 'BTC', vol: 1.0, drift: 1.00, correlation: 0.7 },
      { name: 'ETH', vol: 1.2, drift: 1.00, correlation: 0.8 },
      { name: 'SOL', vol: 1.5, drift: 1.00, correlation: 0.6 }
    ];
  }
  
  getStrategyWeights() {
    const weights = {};
    document.querySelectorAll('.strat-weight').forEach(slider => {
      const strategy = slider.dataset.strat;
      weights[strategy] = parseFloat(slider.value) / 100;
    });
    return weights;
  }
  
  getEconomicEvents() {
    const events = [];
    document.querySelectorAll('.event-row').forEach(row => {
      const impact = row.querySelector('.event-impact');
      let impactValue = 1.0;
      if (impact.classList.contains('impact-high')) impactValue = 1.5;
      else if (impact.classList.contains('impact-medium')) impactValue = 1.2;
      
      const eventName = row.querySelector('div:nth-child(2)').textContent.trim();
      const day = parseInt(row.querySelector('.event-day').value);
      const effect = row.querySelector('.event-effect').value;
      
      events.push({ day, effect, impact: impactValue, name: eventName });
    });
    return events;
  }
  
  getNewsEvents() {
    const events = [];
    document.querySelectorAll('.news-item').forEach(item => {
      const time = item.querySelector('.news-time').textContent;
      const text = item.querySelector('input[type="text"]') ? item.querySelector('input[type="text"]').value : item.querySelector('div:not(.news-time)').textContent;
      events.push({ time, text });
    });
    return events;
  }
  
  getCorrelationMatrix() {
    return {
      'BTC-ETH': parseFloat(getElement('corr-btc-eth').textContent || 0.78),
      'BTC-SOL': parseFloat(getElement('corr-btc-sol').textContent || 0.65),
      'ETH-SOL': parseFloat(getElement('corr-eth-sol').textContent || 0.82)
    };
  }

  getTraderEfficiency(traderId) {
    const val = parseFloat(getElement(`trader${traderId}Efficiency`)?.value || 1);
    return isNaN(val) || val < 0 ? 1 : val;
  }

  getRiskTiersFromUI() {
    const tiers = [];
    document.querySelectorAll('.risk-tier-row').forEach(row => {
      const threshold = parseFloat(row.querySelector('.risk-tier-threshold').value);
      const riskPercent = parseFloat(row.querySelector('.risk-tier-percent').value);
      if (!isNaN(threshold) && !isNaN(riskPercent)) tiers.push({ threshold, riskPercent, maxThreshold: Infinity });
    });
    
    if (tiers.length === 0) {
      return [
        { threshold: 0, maxThreshold: 100, riskPercent: 0.8 },
        { threshold: 100, maxThreshold: 500, riskPercent: 0.08 },
        { threshold: 500, maxThreshold: 1000, riskPercent: 0.04 },
        { threshold: 1000, maxThreshold: 5000, riskPercent: 0.03 },
        { threshold: 5000, maxThreshold: Infinity, riskPercent: 0.02 }
      ];
    }
    
    tiers.sort((a, b) => a.threshold - b.threshold);
    for (let i = 0; i < tiers.length - 1; i++) tiers[i].maxThreshold = tiers[i + 1].threshold;
    tiers[tiers.length - 1].maxThreshold = Infinity;
    return tiers;
  }

  pickAsset(day) {
    const correlatedAssets = this.assets.map(asset => {
      const correlationEffect = 1 + (this.correlationStrength - 0.5) * 0.5;
      return { ...asset, vol: asset.vol * correlationEffect };
    });
    
    const idx = Math.floor(Math.sin(day * 0.314) * correlatedAssets.length + correlatedAssets.length) % correlatedAssets.length;
    return correlatedAssets[idx];
  }

  getPsychologyEffect(streak, capital, peak) {
    const drawdown = peak > 0 ? ((peak - capital) / peak) * 100 : 0;
    if (streak <= -3) {
      const tilt = Math.min(1.5, 1 + (Math.abs(streak) - 2) * 0.3);
      return { multiplier: tilt, reason: 'revenge-tilt' };
    }
    if (streak >= 5) {
      const greed = Math.min(1.4, 1 + (streak - 4) * 0.1);
      return { multiplier: greed, reason: 'over-confidence' };
    }
    if (drawdown > 20 && streak === 0) {
      return { multiplier: 1.2, reason: 'fomo-drawdown' };
    }
    return { multiplier: 1, reason: 'normal' };
  }

  updateTraderFatigue(traderId, isWin) {
    const profile = this.traderProfiles.find(p => p.id === traderId);
    if (!profile) return;
    
    if (isWin) {
      profile.streak = Math.max(0, profile.streak + 1);
    } else {
      profile.streak = Math.min(0, profile.streak - 1);
    }
    
    const delta = isWin ? -0.02 : 0.03;
    profile.fatigue = Math.max(0, Math.min(1, profile.fatigue + delta));
  }

  getTraderFatigueMultiplier(traderId) {
    const profile = this.traderProfiles.find(p => p.id === traderId);
    if (!profile) return 1;
    return 1 / (1 + Math.exp(4 * (profile.fatigue - 0.5)));
  }
  
  getNewsShock(day) {
    const scheduledEvent = this.newsEvents.find(e => {
      const eventDay = parseInt(e.time.split(',')[0].replace('Day ', ''));
      return eventDay === day;
    });
    
    if (scheduledEvent) {
      const severity = 0.5 + Math.random() * 0.5;
      return {
        active: true,
        severity,
        slippage: severity * 0.01 * this.newsSensitivity,
        edgePenalty: severity * 0.15 * this.newsSensitivity,
        reason: `Scheduled news: ${scheduledEvent.text}`
      };
    }
    
    const seed = Math.sin(day * 0.314) * 10000;
    const dice = ((seed % 1000) + 1000) % 1000;
    
    if (dice < 20) {
      const severity = 0.2 + Math.random() * 0.8;
      return {
        active: true,
        severity,
        slippage: severity * 0.01 * this.newsSensitivity,
        edgePenalty: severity * 0.15 * this.newsSensitivity,
        reason: 'Random market shock'
      };
    }
    return { active: false };
  }
  
  getStochasticWinRate(baseRate, volatilityMult, streak, fatigueMult) {
    const noise = (Math.random() - 0.5) * 0.12 * volatilityMult;
    let tilt = 0;
    if (streak <= -3) tilt = -0.05;
    if (streak >= 5) tilt = 0.03;
    const fatiguePenalty = (1 - fatigueMult) * 0.15;
    let rate = baseRate + noise + tilt - fatiguePenalty;
    return Math.max(0.45, Math.min(0.99, rate));
  }

  getBaseProfitability(day, days, drawdown, winStreak, tradeCount) {
    const { initialProfitability, finalProfitability } = this.config;
    const decayMult = this.getDecayMultiplier(day, days, drawdown, winStreak, tradeCount);
    const rawEdge = initialProfitability - (initialProfitability - finalProfitability) * Math.pow(day / days, 1.5);
    return Math.max(finalProfitability, Math.min(0.99, rawEdge * decayMult));
  }
  
  getDecayMultiplier(day, days, drawdown, winStreak, tradeCount) {
    const crowding = Math.min(0.3, tradeCount / 10000);
    let ddDecay = 0;
    if (drawdown > 30) ddDecay = 0.2;
    else if (drawdown > 15) ddDecay = 0.08 + (drawdown - 15) * 0.008;
    const streakBonus = winStreak >= 5 ? -0.05 : (winStreak <= -5 ? 0.05 : 0);
    const cycle = Math.sin(day / (this.config.cyclicalPeriod || 14)) * 0.04;
    return 1 - crowding - ddDecay + streakBonus + cycle;
  }

  getVolatilityMultiplier(day) {
    const regime = this.getRegimeAndVol(day);
    const noise = (Math.random() - 0.5) * 0.2;
    return Math.max(0.4, (regime.vol * this.config.baseVolatility) + noise);
  }
  
  getRegimeAndVol(day) {
    const event = this.economicEvents.find(e => e.day === day);
    if (event) {
      switch (event.effect) {
        case 'bull': return { name: 'bull', vol: 0.8 * event.impact, drift: 1.05 };
        case 'bear': return { name: 'bear', vol: 1.4 * event.impact, drift: 0.85 };
        case 'volatile': return { name: 'volatile', vol: 2.0 * event.impact, drift: 1.00 };
        default: break;
      }
    }
    
    switch (this.config.marketCondition) {
      case 'bull': return { name: 'bull', vol: 0.8, drift: 1.05 };
      case 'bear': return { name: 'bear', vol: 1.4, drift: 0.85 };
      case 'volatile': return { name: 'volatile', vol: 2.0, drift: 1.00 };
      default: return { name: 'normal', vol: 1.1, drift: 1.00 };
    }
  }
  
  getFeeSlip(positionSize) {
    let feeRate = 0.003;
    if (positionSize < 1000) feeRate = 0.003;
    else if (positionSize < 10000) feeRate = 0.002;
    else if (positionSize < 100000) feeRate = 0.0015;
    else feeRate = 0.001;
    
    feeRate = feeRate * (1 + this.priceImpact);
    
    const regime = this.getRegimeAndVol(1);
    const baseSlip = 0.0005;
    const volSlip = Math.min(0.01, 0.0001 * Math.log1p(positionSize / 1000));
    let slipRate = baseSlip + volSlip + (regime.vol - 1) * 0.0002;
    
    slipRate = slipRate / this.liquidityFactor;
    
    return { feeRate, slipRate: Math.max(0.0001, slipRate) };
  }
  
  checkStops(capital, tradePnl) {
    const dd = ((this.peakCapital - capital) / this.peakCapital) * 100;
    if (dd >= this.stopConfig.maxDrawdown * 100) return 'maxDrawdown';
    if (tradePnl <= -this.stopConfig.maxDailyLoss * capital) return 'dailyStop';
    return null;
  }
  
  applyTakeProfitAndTrailingStop(positionSize, openPrice, currentPrice) {
    const takeProfit = this.stopConfig.takeProfit;
    const trailingStop = this.stopConfig.trailingStop;
    
    const profitPercent = ((currentPrice - openPrice) / openPrice) * 100;
    
    if (profitPercent >= takeProfit * 100) { // Take profit is percentage, convert for comparison
      return { close: true, profit: positionSize * takeProfit };
    }
    
    const maxPrice = Math.max(openPrice, currentPrice);
    const trailingStopPrice = maxPrice * (1 - trailingStop); // Trailing stop is percentage
    
    if (currentPrice <= trailingStopPrice) {
      const loss = ((openPrice - currentPrice) / openPrice) * positionSize;
      return { close: true, profit: -loss };
    }
    
    return { close: false, profit: 0 };
  }
  
  generateCandle(prevClose, regime, asset) {
    const drift = asset.drift - 1;
    const vol = asset.vol * regime.vol;
    
    const open = prevClose * (1 + drift * 0.002);
    const noise = (Math.random() - 0.5) * vol * 0.04;
    const high = open * (1 + Math.abs(noise) + Math.random() * vol * 0.02);
    const low = open * (1 - Math.abs(noise) - Math.random() * vol * 0.02);
    const close = open * (1 + noise);
    
    const volume = Math.floor(1000000 * (0.8 + Math.random() * 0.4) * (1 + vol * 0.5));
    
    return { open, high, low, close, volume };
  }
  
  pickStrategy(day) {
    if (Object.keys(this.strategyWeights).length > 0) {
      const totalWeight = Object.values(this.strategyWeights).reduce((sum, weight) => sum + weight, 0);
      if (totalWeight > 0) {
        const rand = Math.random() * totalWeight;
        let cumulative = 0;
        
        for (const [strategy, weight] of Object.entries(this.strategyWeights)) {
          cumulative += weight;
          if (rand <= cumulative) {
            return this.strategies.find(s => s.name === strategy);
          }
        }
      }
    }
    
    const idx = Math.floor(Math.sin(day * 0.5) * this.strategies.length + this.strategies.length) % this.strategies.length;
    return this.strategies[idx];
  }

  calculateTradeOutcome(capital, day, days, traderId, recentPerformance) {
    const strat = this.pickStrategy(day);
    const regime = this.getRegimeAndVol(day);
    const asset = this.pickAsset(day);
    
    const prevClose = asset.priceHistory?.[asset.priceHistory.length - 1]?.close || 10000;
    const candle = this.generateCandle(prevClose, regime, asset);
    
    if (!asset.priceHistory) asset.priceHistory = [];
    asset.priceHistory.push(candle);
    
    const peak = Math.max(this.peakCapital, capital);
    this.peakCapital = peak;
    const streak = (this.consecutiveWins || 0) - (this.consecutiveLosses || 0);
    const psyche = this.getPsychologyEffect(streak, capital, peak);
    
    let riskPercent = this.getCurrentRiskTier(capital).riskPercent * psyche.multiplier;
    if (this.config.riskTolerance === 'conservative') riskPercent *= 0.7;
    else if (this.config.riskTolerance === 'aggressive') riskPercent *= 1.3;
    riskPercent = Math.max(0.01, Math.min(1, riskPercent));
    
    const stratEdge = strat.baseEdge * Math.max(0.5, 1 - (this.tradeLog.length / 10000) * strat.decay);
    const volMult = this.getVolatilityMultiplier(day);
    const fatigue = this.getTraderFatigueMultiplier(traderId);
    const winRate = this.getStochasticWinRate(stratEdge, volMult, streak, fatigue);
    
    let positionSize = capital * riskPercent * this.config.leverage;
    const { feeRate, slipRate } = this.getFeeSlip(positionSize);
    const fee = positionSize * feeRate;
    const slipLoss = positionSize * slipRate;
    positionSize -= (fee + slipLoss);
    
    const basePos = 5 * this.config.leverage;
    const eff = this.getTraderEfficiency(parseInt(traderId.replace('TRADER_', '')));
    const scaled = (positionSize / basePos) * eff;
    
    const news = this.getNewsShock(day);
    let isWin = Math.random() < winRate;
    if (news.active) {
      const shockedRate = Math.max(0.45, winRate - news.edgePenalty);
      isWin = Math.random() < shockedRate;
    }
    
    const openPrice = candle.open;
    const closePrice = candle.close;
    const stopResult = this.applyTakeProfitAndTrailingStop(scaled, openPrice, closePrice);
    
    let result;
    if (stopResult.close) {
      result = stopResult.profit;
      isWin = result > 0;
    } else {
      result = isWin ? scaled * volMult : -scaled * volMult;
    }
    
    const finalSlip = Math.random() * 0.005;
    result *= (1 - finalSlip);
    const newCapital = Math.max(0.01, capital + result);
    
    if (isWin) {
      this.consecutiveWins++;
      this.consecutiveLosses = 0;
    } else {
      this.consecutiveLosses++;
      this.consecutiveWins = 0;
    }
    
    this.updateTraderFatigue(traderId, isWin);
    
    this.tradeLog.push({
      day, 
      trader: traderId, 
      strategy: strat.name, 
      positionSize, 
      result, 
      isWin,
      fee, 
      slippage: slipLoss + finalSlip * positionSize,
      newCapital, 
      candle, 
      asset: asset.name,
      note: (psyche.reason !== 'normal') ? `Psychology: ${psyche.reason}` : (news.active ? news.reason : '')
    });
    
    if (this.tradeLog.length > 10000) this.tradeLog = this.tradeLog.slice(-10000);
    
    this.phaseMemory.push(isWin ? 1 : 0);
    if (this.phaseMemory.length > 20) this.phaseMemory.shift();
    
    return { positionSize, tradeResult: result, isWin, profitability: winRate, fee, slippage: finalSlip, newCapital };
  }
  
  getCurrentRiskTier(capital) {
    const safe = Math.min(capital, 1e12);
    const tier = this.riskTiers.find(t => safe >= t.threshold && safe < t.maxThreshold);
    return tier || { riskPercent: 0.05, useFullCapital: false };
  }
  
  simulate(days, updateProgress) {
    let capital = this.config.startingCapital;
    let totalTrades = 0, winningTrades = 0;
    let dailyResults = [];
    let recentWindow = [];
    let totalFees = 0, totalSlippage = 0;
    let peak = capital, maxDD = 0;
    const returns = [];
    const hoursPerTrader = this.config.tradingHours / this.config.numTraders;
    
    for (let day = 1; day <= days; day++) {
      if (updateProgress) updateProgress(Math.floor(day / days * 100));
      
      const dayStart = capital;
      const recent = recentWindow.length ? recentWindow.reduce((a,b)=>a+b,0)/recentWindow.length : 0.65;
      
      for (const profile of this.traderProfiles) {
        const traderId = profile.id;
        const eff = this.getTraderEfficiency(parseInt(traderId.replace('TRADER_', ''))) * this.getTraderFatigueMultiplier(traderId);
        
        for (let h = 0; h < hoursPerTrader; h++) {
          if (capital <= 0.01) break;
          
          const outcome = this.calculateTradeOutcome(
            capital,
            day,
            days,
            traderId,
            recent
          );
          
          capital = outcome.newCapital;
          totalTrades++;
          totalFees += outcome.fee;
          totalSlippage += Math.abs(outcome.tradeResult) * outcome.slippage;
          
          if (outcome.isWin) winningTrades++;
          recentWindow.push(outcome.isWin ? 1 : 0);
          if (recentWindow.length > 100) recentWindow.shift();
        }
      }
      
      if (this.config.compoundFrequency === 'daily') capital *= 1.0002;
      capital = Math.max(0.01, capital);
      
      const dailyGrowth = ((capital - dayStart) / dayStart) * 100;
      returns.push(isNaN(dailyGrowth) ? 0 : dailyGrowth / 100);
      if (capital > peak) peak = capital;
      const dd = ((peak - capital) / peak) * 100;
      if (dd > maxDD) maxDD = dd;
      dailyResults.push({ day, capital, dailyGrowth });
    }
    
    const { sharpeRatio, sortinoRatio, profitFactor } = this.calculateAdvancedMetrics(returns);
    const totalGrowth = ((capital - this.config.startingCapital) / this.config.startingCapital) * 100;
    const years = Math.max(days, 1) / 365;
    const annualized = totalGrowth === 0 ? 0 : (Math.pow(1 + totalGrowth / 100, 1 / years) - 1);
    const calmar = maxDD > 0 ? annualized / (maxDD / 100) : 0;
    
    return {
      finalCapital: capital,
      totalTrades,
      winningTrades,
      actualWinRate: totalTrades ? winningTrades / totalTrades : 0,
      dailyResults,
      totalGrowth,
      totalFees,
      totalSlippage,
      maxDrawdown: maxDD,
      sharpeRatio,
      sortinoRatio,
      profitFactor,
      calmarRatio: calmar,
      tradeLog: this.tradeLog,
      config: this.config
    };
  }
  
  calculateAdvancedMetrics(returns) {
    if (!Array.isArray(returns) || returns.length === 0) return { sharpeRatio: 0, sortinoRatio: 0, profitFactor: 0 };
    const avg = returns.reduce((a,b)=>a+b,0) / returns.length;
    const var_ = returns.reduce((a,b)=>a+Math.pow(b-avg,2),0) / returns.length;
    const std = Math.sqrt(Math.max(var_,0));
    const rf = 0.02 / 365;
    const sharpe = std && isFinite(std) ? (avg - rf) / std * Math.sqrt(365) : 0;
    const downside = returns.filter(r=>r<0);
    const dVar = downside.length ? downside.reduce((a,b)=>a+b*b,0)/downside.length : 0;
    const dStd = Math.sqrt(Math.max(dVar,0));
    const sortino = dStd && isFinite(dStd) ? (avg - rf) / dStd * Math.sqrt(365) : 0;
    const wins = returns.filter(r=>r>0).reduce((a,b)=>a+b,0);
    const losses = Math.abs(returns.filter(r=>r<0).reduce((a,b)=>a+b,0));
    const pf = losses ? wins / losses : (wins ? Infinity : 0);
    return { sharpeRatio: isFinite(sharpe)?sharpe:0, sortinoRatio: isFinite(sortino)?sortino:0, profitFactor: isFinite(pf)?pf:0 };
  }
}

// Utility to get current configuration from UI elements
export function getConfigFromUI() {
  return {
    startingCapital: parseFloat(getElement('startingCapital').value),
    leverage: parseInt(getElement('leverage').value),
    tradingHours: parseInt(getElement('tradingHours').value) || 8,
    numTraders: parseInt(getElement('numTraders').value),
    initialProfitability: parseFloat(getElement('initialProfitability').value) / 100,
    finalProfitability: parseFloat(getElement('finalProfitability').value) / 100,
    marketCondition: getElement('marketCondition').value,
    compoundFrequency: getElement('compoundFrequency').value,
    riskTolerance: getElement('riskTolerance').value,
    baseVolatility: parseFloat(getElement('baseVolatility').value) || 1.0,
    cyclicalPeriod: parseFloat(getElement('cyclicalPeriod').value) || 14
  };
}
