const mongoose = require('mongoose');

/**
 * ScanRecord — one document per X-ray analysis.
 * Saved by uploadController.js after getting the AI response.
 */
const scanRecordSchema = new mongoose.Schema(
  {
    userId: {
      type: mongoose.Schema.Types.ObjectId,
      ref: 'User',
      required: true,
      index: true,
    },
    patientId: { type: String, default: null },       // optional patient identifier
    filename:  { type: String, required: true },       // original upload filename
    result: {
      type: String,
      enum: ['TB_DETECTED', 'NORMAL', 'INCONCLUSIVE'],
      required: true,
    },
    confidence:    { type: Number, min: 0, max: 1, required: true },
    originalImageBase64: { type: String, default: null },
    gradcamBase64: { type: String, default: null },    // heatmap
    overlayBase64: { type: String, default: null },    // overlay
    gradcamUrl:    { type: String, default: null },    // set if you store to disk/S3
    modelVersion:  { type: String, default: 'ResNet-50 v1' },
    processingMs:  { type: Number, default: null },    // inference latency
    rawResponse:   { type: mongoose.Schema.Types.Mixed, default: null }, // full AI payload
  },
  { timestamps: true }
);

scanRecordSchema.index({ userId: 1, createdAt: -1 });

module.exports = mongoose.model('ScanRecord', scanRecordSchema);