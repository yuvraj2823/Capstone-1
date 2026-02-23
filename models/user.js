const mongoose = require('mongoose');

const UserSchema = new mongoose.Schema({
  name: String,
  email: String,
  password: String
});

// Prevent OverwriteModelError
module.exports =
  mongoose.models.User || mongoose.model('User', UserSchema);