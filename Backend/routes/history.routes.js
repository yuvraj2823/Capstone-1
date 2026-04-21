const express  = require('express');
const router   = express.Router();
const { authenticate } = require('../middlewares/auth');   // your existing JWT middleware
const {
  getHistory,
  getReport,
} = require('../controllers/historyController');

// GET /api/history  — paginated scan history for the logged-in user
router.get('/',         authenticate, getHistory);

// GET /api/history/:id/report  — stream a PDF diagnostic report
router.get('/:id/report', authenticate, getReport);

module.exports = router;