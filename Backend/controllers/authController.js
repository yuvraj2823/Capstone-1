const jwt = require('jsonwebtoken');
const User = require('../models/User');
const { createAuditLog } = require('../services/auditService');
const logger = require('../config/logger');

/**
 * Generates a signed JWT for a given user ID.
 */
function signToken(userId) {
  return jwt.sign(
    { id: userId },
    process.env.JWT_SECRET,
    { expiresIn: process.env.JWT_EXPIRES_IN || '7d' }
  );
}

/**
 * POST /api/auth/signup
 */
async function signup(req, res, next) {
  try {
    const { username, email, password } = req.body;

    if (!username || !email || !password) {
      return res.status(400).json({ success: false, error: 'Username, email, and password are required.' });
    }

    const user = await User.create({ username, email, password });
    const token = signToken(user._id);

    await createAuditLog({
      userId: user._id,
      username: user.username,
      action: 'SIGNUP',
      endpoint: '/api/auth/signup',
      method: 'POST',
      statusCode: 201,
      ipAddress: req.ip,
      userAgent: req.headers['user-agent'],
    });

    logger.info(`New user registered: ${email}`);

    res.status(201).json({
      success: true,
      token,
      user: {
        id: user._id,
        username: user.username,
        email: user.email,
        role: user.role,
      },
    });
  } catch (err) {
    next(err);
  }
}

/**
 * POST /api/auth/login
 */
async function login(req, res, next) {
  try {
    const { email, password } = req.body;

    if (!email || !password) {
      return res.status(400).json({ success: false, error: 'Email and password are required.' });
    }

    const user = await User.findOne({ email }).select('+password');
    if (!user || !(await user.comparePassword(password))) {
      await createAuditLog({
        username: email,
        action: 'AUTH_FAILURE',
        endpoint: '/api/auth/login',
        method: 'POST',
        statusCode: 401,
        ipAddress: req.ip,
        userAgent: req.headers['user-agent'],
      });
      return res.status(401).json({ success: false, error: 'Invalid email or password.' });
    }

    if (!user.isActive) {
      return res.status(403).json({ success: false, error: 'Account has been deactivated.' });
    }

    user.lastLogin = new Date();
    await user.save({ validateBeforeSave: false });

    const token = signToken(user._id);

    await createAuditLog({
      userId: user._id,
      username: user.username,
      action: 'LOGIN',
      endpoint: '/api/auth/login',
      method: 'POST',
      statusCode: 200,
      ipAddress: req.ip,
      userAgent: req.headers['user-agent'],
    });

    res.status(200).json({
      success: true,
      token,
      user: {
        id: user._id,
        username: user.username,
        email: user.email,
        role: user.role,
      },
    });
  } catch (err) {
    next(err);
  }
}

/**
 * GET /api/auth/me
 */
async function getMe(req, res) {
  res.status(200).json({ success: true, user: req.user });
}

module.exports = { signup, login, getMe };
