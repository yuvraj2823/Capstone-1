const axios = require('axios');
const CircuitBreaker = require('opossum');
const logger = require('../config/logger');

const AI_SERVICE_URL = process.env.AI_SERVICE_URL || 'http://ai-service:8000';
const TIMEOUT_MS = parseInt(process.env.AI_TIMEOUT_MS || '30000', 10);

/**
 * Core function that calls the AI service.
 * @param {string} imageBase64  - base64 encoded image string
 * @param {string} mimeType     - image MIME type
 * @returns {Promise<{prediction: number, heatmap: string, overlay: string}>}
 */
async function callAIServiceRaw(imageBase64, mimeType = 'image/jpeg') {
  const response = await axios.post(
    `${AI_SERVICE_URL}/predict`,
    { image: imageBase64, mime_type: mimeType },
    {
      timeout: TIMEOUT_MS,
      headers: {
        'Content-Type': 'application/json',
        Accept: 'application/json',
      },
    }
  );
  return response.data;
}

// ─── Circuit Breaker Configuration ────────────────────────────────────────────
const cbOptions = {
  timeout: TIMEOUT_MS,
  errorThresholdPercentage: 50,
  resetTimeout: 30000, // 30s before retrying after open
  name: 'ai-service-breaker',
  rollingCountTimeout: 10000,
  rollingCountBuckets: 10,
};

const breaker = new CircuitBreaker(callAIServiceRaw, cbOptions);

breaker.on('open', () => logger.warn('⚡ Circuit breaker OPEN — AI service calls blocked'));
breaker.on('halfOpen', () => logger.info('⚡ Circuit breaker HALF-OPEN — Testing AI service'));
breaker.on('close', () => logger.info('✅ Circuit breaker CLOSED — AI service healthy'));
breaker.on('fallback', (result) => logger.warn('Circuit breaker fallback triggered:', result));
breaker.fallback(() => {
  const err = new Error('AI service is temporarily unavailable (circuit open).');
  err.code = 'EOPENBREAKER';
  throw err;
});

/**
 * Public service method — calls AI service with circuit breaker protection.
 */
async function predictTB(imageBase64, mimeType) {
  logger.info('Forwarding image to AI service for prediction...');
  const result = await breaker.fire(imageBase64, mimeType);
  logger.info(`AI service responded. Prediction score: ${result.prediction}`);
  return result;
}

/**
 * Returns current circuit breaker health stats.
 */
function getBreakerStats() {
  return breaker.stats;
}

module.exports = { predictTB, getBreakerStats };
