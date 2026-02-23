const express = require('express');
const router = express.Router();
const User = require('../models/user');
const bcrypt = require('bcrypt');


// =======================
// LOGIN PAGE
// =======================

// Main login page
router.get('/', (req, res) => {
  res.render('login');
});

// ⭐ Login page via /login URL
router.get('/login', (req, res) => {
  res.render('login');
});


// =======================
// SIGNUP PAGE
// =======================
router.get('/signup', (req, res) => {
  res.render('signup');
});


// =======================
// SIGNUP (CREATE USER)
// =======================
router.post('/signup', async (req, res) => {
  try {
    const { name, email, password } = req.body;

    const existing = await User.findOne({ email });
    if (existing) return res.send('User already exists');

    const hashedPassword = await bcrypt.hash(password, 10);

    const user = new User({
      name,
      email,
      password: hashedPassword
    });

    await user.save();

    res.redirect('/login'); // go to login after signup
  } catch (err) {
    console.error(err);
    res.send('Signup error');
  }
});


// =======================
// LOGIN (AUTHENTICATE)
// =======================
router.post('/login', async (req, res) => {
  try {
    const { email, password } = req.body;

    const user = await User.findOne({ email });
    if (!user) return res.send('User not found');

    const match = await bcrypt.compare(password, user.password);
    if (!match) return res.send('Wrong password');

    req.login(user, (err) => {
      if (err) return res.send('Login error');
      res.redirect('/dashboard');
    });

  } catch (err) {
    console.error(err);
    res.send('Login error');
  }
});


// =======================
// DASHBOARD (PROTECTED)
// =======================
router.get('/dashboard', (req, res) => {
  if (req.isAuthenticated()) {
    res.render('dashboard', { user: req.user });
  } else {
    res.redirect('/login');
  }
});


// =======================
// LOGOUT
// =======================
router.get('/logout', (req, res) => {
  req.logout(() => res.redirect('/login'));
});


module.exports = router;