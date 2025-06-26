import os
import numpy as np
from PIL import Image
import logging
from typing import Optional, Tuple, List, Dict
import io
import glob
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)


def predict_image_from_bytes(image_bytes: bytes, model) -> Tuple[Optional[float], Optional[str]]:
    """
    Predict bathroom probability from image bytes.
    
    Args:
        image_bytes: Raw image bytes
        model: Loaded Keras model
        
    Returns:
        Tuple of (probability, error_message)
        If successful, returns (probability, None)
        If failed, returns (None, error_message)
    """
    try:
        # Load image from bytes
        img = Image.open(io.BytesIO(image_bytes))
        
        # Convert to RGB if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Resize to target size (224, 224)
        img = img.resize((224, 224))
        
        # Convert to array and preprocess
        img_array = np.array(img, dtype=np.float32)
        img_array = np.expand_dims(img_array, axis=0)
        
        # Make prediction
        prediction = model.predict(img_array, verbose=0)
        probability = float(prediction[0][0])
        
        return probability, None
        
    except Exception as e:
        error_msg = f"Prediction failed: {str(e)}"
        logger.warning(error_msg)
        return None, error_msg


def predict_image_from_file(image_path: str, model) -> Tuple[Optional[float], Optional[str]]:
    """
    Predict bathroom probability from image file.
    
    Args:
        image_path: Path to image file
        model: Loaded Keras model
        
    Returns:
        Tuple of (probability, error_message)
    """
    try:
        with open(image_path, 'rb') as f:
            image_bytes = f.read()
        return predict_image_from_bytes(image_bytes, model)
    except Exception as e:
        error_msg = f"Failed to read image file: {str(e)}"
        logger.warning(error_msg)
        return None, error_msg


def create_predicted_filename(base_path: str, probability: float, index: int) -> str:
    """
    Create filename with probability prefix.
    
    Args:
        base_path: Base directory path
        probability: Prediction probability
        index: Image index (1-based)
        
    Returns:
        Full path to predicted image file
    """
    prob_str = f"{probability:.4f}"
    return os.path.join(base_path, f"{prob_str}_{index}.jpg")


def create_unpredicted_filename(base_path: str, index: int) -> str:
    """
    Create filename without prediction prefix.
    
    Args:
        base_path: Base directory path
        index: Image index (1-based)
        
    Returns:
        Full path to unpredicted image file
    """
    return os.path.join(base_path, f"{index}.jpg")


def process_single_listing_images(args) -> Tuple[str, int, int]:
    """
    Process all images for a single listing.
    
    Args:
        args: Tuple of (offer_id, offer_dir, model)
        
    Returns:
        Tuple of (offer_id, processed_count, error_count)
    """
    offer_id, offer_dir, model = args
    processed_count = 0
    error_count = 0
    
    # Find all unpredicted images (those without probability prefix)
    image_files = []
    for filename in os.listdir(offer_dir):
        if filename.endswith('.jpg'):
            # Check if this is an unpredicted image (no underscore in filename)
            if '_' not in filename:
                # This is an unpredicted image (e.g., "1.jpg", "2.jpg", etc.)
                try:
                    # Extract index from filename
                    index = int(filename.split('.')[0])
                    image_files.append((filename, index))
                except ValueError:
                    continue
    
    if not image_files:
        logger.info(f"No unpredicted images found for offer {offer_id}")
        return offer_id, 0, 0
    
    logger.info(f"🤖 Processing {len(image_files)} images for offer {offer_id}")
    
    for filename, index in image_files:
        old_path = os.path.join(offer_dir, filename)
        
        try:
            # Predict
            probability, error_msg = predict_image_from_file(old_path, model)
            
            if probability is not None:
                # Create new filename with prediction
                new_path = create_predicted_filename(offer_dir, probability, index)
                
                # Rename file
                os.rename(old_path, new_path)
                logger.debug(f"✅ Renamed {filename} to {os.path.basename(new_path)}")
                processed_count += 1
            else:
                logger.warning(f"❌ Failed to predict {filename}: {error_msg}")
                error_count += 1
                
        except Exception as e:
            logger.error(f"❌ Error processing {filename}: {str(e)}")
            error_count += 1
    
    logger.info(f"✅ Offer {offer_id}: {processed_count} processed, {error_count} errors")
    return offer_id, processed_count, error_count


def predict_images_batch(
    listings: List[Dict],
    image_dir: str,
    model,
    max_workers: int = 4
) -> Dict[str, int]:
    """
    Predict bathroom probabilities for all downloaded images in batch.
    
    Args:
        listings: List of listings with offer_id
        image_dir: Base directory containing offer subdirectories
        model: Loaded Keras model
        max_workers: Number of parallel workers
        
    Returns:
        Dictionary with statistics
    """
    logger.info(f"🤖 Starting batch prediction for {len(listings)} listings")
    
    # Prepare tasks
    tasks = []
    for listing in listings:
        offer_id = str(listing.get('offer_id'))
        offer_dir = os.path.join(image_dir, offer_id)
        
        if os.path.exists(offer_dir):
            tasks.append((offer_id, offer_dir, model))
    
    if not tasks:
        logger.warning("No offer directories found for prediction")
        return {"total_offers": 0, "total_processed": 0, "total_errors": 0}
    
    total_processed = 0
    total_errors = 0
    successful_offers = 0
    
    # Process in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_task = {
            executor.submit(process_single_listing_images, task): task[0]
            for task in tasks
        }
        
        for future in as_completed(future_to_task):
            offer_id = future_to_task[future]
            try:
                _, processed, errors = future.result()
                total_processed += processed
                total_errors += errors
                if processed > 0:
                    successful_offers += 1
            except Exception as e:
                logger.error(f"Failed to process offer {offer_id}: {str(e)}")
                total_errors += 1
    
    logger.info(
        f"✅ Batch prediction complete: {successful_offers}/{len(tasks)} offers, "
        f"{total_processed} images processed, {total_errors} errors"
    )
    
    return {
        "total_offers": len(tasks),
        "successful_offers": successful_offers,
        "total_processed": total_processed,
        "total_errors": total_errors
    }