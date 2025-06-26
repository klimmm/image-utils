# Image Processing Utilities

Image processing utilities for real estate applications.

## Features

- Image downloading from URLs
- Image deduplication using perceptual hashing
- Image quality prediction using TensorFlow model
- Batch processing capabilities

## Installation

```bash
pip install tensorflow pillow numpy scikit-learn imagehash
```

## Usage

### As a library

```python
from image_filter import prefilter_listings_for_download
from download_images import download_images
from image_dedup import dedupe_images
from predict_image import predict_images_batch
from load_model import load_model

# Load the model
model = load_model('best_model.h5')

# Filter listings
filtered = prefilter_listings_for_download(listings_data)

# Download images
download_images(filtered, output_dir)

# Deduplicate
dedupe_images(output_dir)

# Predict quality
predictions = predict_images_batch(output_dir, model)
```

### As a GitHub Action

This repository can be used as a reusable workflow:

```yaml
jobs:
  process:
    uses: klimmm/image-utils/.github/workflows/process-images.yml@main
    with:
      image_urls_json: ${{ needs.previous-job.outputs.urls }}
      repository: ${{ github.repository }}
    secrets:
      token: ${{ secrets.GITHUB_TOKEN }}
```

## Model

The `best_model.h5` file contains a pre-trained TensorFlow model for image quality assessment.