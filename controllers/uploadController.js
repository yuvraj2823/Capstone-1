const sharp = require('sharp');
const { predictTB } = require('../services/aiService');
const { createAuditLog } = require('../services/auditService');
const logger = require('../config/logger');

/**
 * POST /api/upload
 * Accepts a multipart/form-data image upload.
 * Preprocesses the image (strips EXIF), sends to AI service, returns results.
 */
async function uploadImage(req, res, next) {
  const start = Date.now();
  try {
    if (!req.file) {
      return res.status(400).json({ success: false, error: 'No image file uploaded.' });
    }

    const { mimetype, buffer, size } = req.file;
    const allowedMimes = ['image/jpeg', 'image/png', 'image/webp', 'image/bmp'];

    if (!allowedMimes.includes(mimetype)) {
      return res.status(415).json({
        success: false,
        error: `Unsupported image format. Allowed: ${allowedMimes.join(', ')}`,
      });
    }

    if (size > 10 * 1024 * 1024) {
      return res.status(413).json({ success: false, error: 'Image must be smaller than 10 MB.' });
    }

    logger.info(`Image received: ${size} bytes, type: ${mimetype}`);

    // Strip EXIF metadata and convert to JPEG via sharp
    const processedBuffer = await sharp(buffer)
      .removeAlpha()
      .toFormat('jpeg', { quality: 90 })
      .withMetadata({ exif: {} }) // clear EXIF
      .toBuffer();

    const imageBase64 = processedBuffer.toString('base64');

    // Call AI service via circuit breaker
    const aiResult = await predictTB(imageBase64, 'image/jpeg');

    const duration = Date.now() - start;

    await createAuditLog({
      userId: req.user?._id,
      username: req.user?.username || 'anonymous',
      action: 'PREDICTION_REQUEST',
      endpoint: '/api/upload',
      method: 'POST',
      statusCode: 200,
      ipAddress: req.ip,
      userAgent: req.headers['user-agent'],
      metadata: { prediction: aiResult.prediction },
      duration,
    });

    res.status(200).json({
      success: true,
      data: {
        prediction: aiResult.prediction,
        heatmap: aiResult.heatmap,
        overlay: aiResult.overlay,
        label: aiResult.prediction > 0.5 ? 'TB Positive' : 'TB Negative',
        confidence: Math.round(Math.max(aiResult.prediction, 1 - aiResult.prediction) * 100),
        processingTimeMs: duration,
      },
    });
  } catch (err) {
    next(err);
  }
}

module.exports = { uploadImage };
