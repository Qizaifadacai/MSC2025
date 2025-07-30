import os
import numpy as np
from PIL import Image
import glob
import matplotlib.pyplot as plt
import cv2
from scipy import ndimage
from skimage import measure, segmentation, filters
from skimage.feature import peak_local_max
from skimage.segmentation import watershed

def add_scale_bar(image_array, pixel_per_um=1004/770, scale_length_um=100, bar_thickness=8, margin=40, font_size=36):
    from PIL import ImageDraw, ImageFont

    img = Image.fromarray(image_array)
    draw = ImageDraw.Draw(img)

    scale_length_pixels = int(scale_length_um * pixel_per_um)
    width, height = img.size

    bar_x_start = width - scale_length_pixels - margin-40
    bar_x_end = width - margin-40
    bar_y_start = height - margin - bar_thickness - 45
    bar_y_end = height - margin - 45

    draw.rectangle([bar_x_start, bar_y_start, bar_x_end, bar_y_end], fill='white')
    text = f"{scale_length_um} μm"

    # 字号缩小2.5
    scaled_font_size = int(font_size - 2.5)
    try:
        font = ImageFont.truetype("/Library/Fonts/Arial Bold.ttf", scaled_font_size)
    except:
        try:
            font = ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf", scaled_font_size)
        except:
            try:
                font = ImageFont.truetype("/Library/Fonts/Arial.ttf", scaled_font_size)
            except:
                try:
                    font = ImageFont.load_default()
                except:
                    font = None

    if font:
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
    else:
        text_width = len(text) * 8
        text_height = 14

    text_x = bar_x_start + (scale_length_pixels - text_width) // 2
    text_y = bar_y_start - text_height - 15

    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            if dx != 0 or dy != 0:
                draw.text((text_x + dx, text_y + dy), text, fill='black', font=font)
    draw.text((text_x, text_y), text, fill='white', font=font)

    return np.array(img)

def count_cells_in_channel(image_array, channel_name, min_cell_size=150, max_cell_size=2000):
    """
    Count individual cells in a single channel using different strategies for red and green.
    
    Red channel: Count all connected red regions regardless of shape
    Green channel: Detect bright elliptical nuclei within green regions
    
    Args:
        image_array: 2D numpy array of the channel
        channel_name: 'green' or 'red' for display purposes
        min_cell_size: minimum area of a cell in pixels
        max_cell_size: maximum area of a cell in pixels
    
    Returns:
        cell_count: number of detected cells
        labeled_image: image with cell boundaries
        cell_mask: binary mask of detected cells
    """
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(image_array, (3, 3), 0)
    
    if channel_name.lower() == 'red':
        # RED CHANNEL: Count all connected red regions
        # Use a lower threshold to capture all red areas
        threshold_value = filters.threshold_otsu(blurred) * 0.7  # More sensitive threshold
        binary = blurred > threshold_value
        
        # Remove very small noise but keep larger connected regions
        binary = ndimage.binary_opening(binary, structure=np.ones((2, 2)))
        binary = ndimage.binary_closing(binary, structure=np.ones((5, 5)))  # Close gaps
        
        # Label connected components
        labels, num_features = ndimage.label(binary)
        
        # Filter by size and count
        filtered_labels = np.zeros_like(labels)
        cell_count = 0
        
        for i in range(1, num_features + 1):
            region_mask = labels == i
            region_size = np.sum(region_mask)
            
            if min_cell_size <= region_size <= max_cell_size:
                cell_count += 1
                filtered_labels[region_mask] = cell_count
        
        # Create boundary image
        boundaries = segmentation.find_boundaries(filtered_labels, mode='thick')
        
    else:  # GREEN CHANNEL
        # GREEN CHANNEL: Detect bright elliptical nuclei within green regions
        
        # First, find the general green regions with much lower threshold
        threshold_low = filters.threshold_otsu(blurred) * 0.3  # Much more sensitive for green background
        green_regions = blurred > threshold_low
        
        # Clean up green regions with smaller kernels to preserve detail
        green_regions = ndimage.binary_opening(green_regions, structure=np.ones((2, 2)))
        green_regions = ndimage.binary_closing(green_regions, structure=np.ones((4, 4)))
        
        # Now find bright nuclei within green regions with lower threshold
        # Use a much lower percentile to catch more nuclei
        if np.any(green_regions):
            threshold_high = np.percentile(blurred[green_regions], 60)  # Top 40% instead of top 20%
        else:
            threshold_high = np.percentile(blurred, 60)
        
        bright_nuclei = blurred > threshold_high
        
        # Only keep nuclei that are within green regions
        bright_nuclei = bright_nuclei & green_regions
        
        # Remove very small noise but preserve small nuclei
        bright_nuclei = ndimage.binary_opening(bright_nuclei, structure=np.ones((1, 1)))
        
        # Use distance transform and watershed for separating touching nuclei
        distance = ndimage.distance_transform_edt(bright_nuclei)
        
        # Find local maxima as nucleus centers with smaller min_distance for closer nuclei
        local_maxima = peak_local_max(distance, min_distance=8, threshold_abs=1)  # Reduced from 15 to 8
        markers = np.zeros_like(distance, dtype=int)
        if len(local_maxima) > 0:
            markers[tuple(local_maxima.T)] = np.arange(1, len(local_maxima) + 1)
        
        # Watershed segmentation on the bright nuclei
        if len(local_maxima) > 0:
            labels = watershed(-distance, markers, mask=bright_nuclei)
        else:
            labels = np.zeros_like(bright_nuclei, dtype=int)
        
        # Filter nuclei by size and shape with more lenient criteria
        filtered_labels = np.zeros_like(labels)
        cell_count = 0
        
        for region in measure.regionprops(labels):
            # Much smaller minimum size and more lenient size range
            if min_cell_size/8 <= region.area <= max_cell_size/3:  # Even smaller nuclei allowed
                # More lenient shape filtering
                if region.eccentricity < 0.95:  # Allow very elongated shapes
                    cell_count += 1
                    filtered_labels[labels == region.label] = cell_count
        
        # Create boundary image
        boundaries = segmentation.find_boundaries(filtered_labels, mode='thick')
    
    return cell_count, filtered_labels, boundaries

def create_cell_analysis_image(green_array, red_array, green_labels, red_labels, green_boundaries, red_boundaries):
    """
    Create a composite image showing cell detection results with different visualization for each channel.
    """
    height, width = green_array.shape
    
    # Create RGB output image
    result_img = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Normalize arrays to 0-255 range
    green_norm = cv2.normalize(green_array, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    red_norm = cv2.normalize(red_array, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # Set base image (dimmed)
    result_img[:, :, 1] = green_norm // 4  # Very dim green background
    result_img[:, :, 0] = red_norm // 4    # Very dim red background
    
    # For red channel: highlight entire detected regions
    red_regions = red_labels > 0
    result_img[red_regions, 0] = 200  # Bright red for detected red regions
    result_img[red_regions, 1] = 0
    result_img[red_regions, 2] = 0
    
    # For green channel: highlight detected nuclei with bright green
    green_nuclei = green_labels > 0
    result_img[green_nuclei, 1] = 255  # Bright green for detected nuclei
    result_img[green_nuclei, 0] = 0
    result_img[green_nuclei, 2] = 0
    
    # Draw boundaries
    # Red boundaries in bright red
    result_img[red_boundaries, 0] = 255
    result_img[red_boundaries, 1] = 0
    result_img[red_boundaries, 2] = 0
    
    # Green boundaries in bright green (for nuclei)
    result_img[green_boundaries, 1] = 255
    result_img[green_boundaries, 0] = 0
    result_img[green_boundaries, 2] = 0
    
    # Where boundaries overlap, use yellow
    overlap = green_boundaries & red_boundaries
    result_img[overlap, 0] = 255  # Red
    result_img[overlap, 1] = 255  # Green
    result_img[overlap, 2] = 0    # Blue = 0 (makes yellow)
    
    return result_img

def count_cells_in_rgb_image(rgb_image):
    """
    Count live and dead cells in RGB fluorescence image using advanced image processing.
    
    Improvements:
    1. More lenient fixed threshold instead of OTSU for detecting dim cells
    2. Enhanced morphological operations to connect cell regions
    3. Improved watershed seeding with peak_local_max
    
    Live cells: Irregular spindle or polygonal shaped green fluorescent regions
    Dead cells: Small, bright red fluorescent dots, usually circular
    
    Args:
        rgb_image: RGB image array (height, width, 3)
    
    Returns:
        live_count: number of live cells (green)
        dead_count: number of dead cells (red)
        annotated_img: image with detected cells outlined
        green_mask: final green cell mask
        red_mask: final red cell mask
    """
    # Convert RGB to BGR for OpenCV processing
    bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    
    # Extract individual channels
    r_channel = rgb_image[:, :, 0]
    g_channel = rgb_image[:, :, 1]
    b_channel = rgb_image[:, :, 2]
    
    # === IMPROVED GREEN CHANNEL PROCESSING (Live Cells) ===
    print("    Processing green channel with improved detection...")
    
    # Step 1: Use original green channel with Gaussian blur
    blurred_green = cv2.GaussianBlur(g_channel, (5, 5), 0)
    
    # Step 2: Apply a low fixed threshold instead of OTSU
    # This is more lenient to capture dim cells
    threshold_value = 40  # Low fixed threshold to capture dim cells
    binary_mask = (blurred_green > threshold_value).astype(np.uint8) * 255
    
    # Step 3: Enhanced morphological operations
    # Remove small noise
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel_open)
    
    # Fill holes and connect fragmented cell regions with a larger kernel
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, kernel_close)
    
    # Save the initial mask for visualization
    initial_mask = cleaned_mask.copy()
    
    # Step 4: Apply improved watershed algorithm to separate overlapping cells
    live_count, separated_cells_mask = apply_watershed_segmentation(cleaned_mask, blurred_green)
    
    # === RED CHANNEL PROCESSING (Dead Cells) - 提高阈值和面积范围 ===
    print("    Processing red channel...")

    enhanced_red = r_channel.astype(np.float32)
    enhanced_red = enhanced_red - 0.3 * g_channel.astype(np.float32) - 0.3 * b_channel.astype(np.float32)
    enhanced_red = np.clip(enhanced_red, 0, 255).astype(np.uint8)

    # 提高红色细胞识别阈值
    blurred_red = cv2.GaussianBlur(enhanced_red, (3, 3), 0)
    # 原代码：_, red_thresh = cv2.threshold(blurred_red, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # 修改为：OTSU基础上加一个偏移
    otsu_val, _ = cv2.threshold(blurred_red, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # 增加阈值偏移（如+20），可根据实际效果调整
    _, red_thresh = cv2.threshold(blurred_red, min(otsu_val+20,255), 255, cv2.THRESH_BINARY)

    kernel_open_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    red_mask = cv2.morphologyEx(red_thresh, cv2.MORPH_OPEN, kernel_open_small)

    red_contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    dead_cells = []
    # 提高面积范围，减少误识别
    min_dead_cell_area = 20   # 原为5，适当提高
    max_dead_cell_area = 120  # 原为200，适当减小

    filtered_red_mask = np.zeros_like(red_mask)
    for contour in red_contours:
        area = cv2.contourArea(contour)
        if min_dead_cell_area <= area <= max_dead_cell_area:
            dead_cells.append(contour)
            cv2.fillPoly(filtered_red_mask, [contour], 255)

    dead_count = len(dead_cells)
    
    # === RESULT VISUALIZATION ===
    print(f"    Detected {live_count} live cells and {dead_count} dead cells")
    
    # Create annotated image
    annotated_img = rgb_image.copy()
    annotated_bgr = cv2.cvtColor(annotated_img, cv2.COLOR_RGB2BGR)
    
    # Draw initial mask contours in blue for debugging
    initial_contours, _ = cv2.findContours(initial_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(annotated_bgr, initial_contours, -1, (255, 0, 0), 1)  # Blue contours for initial mask
    
    # Draw separated live cells with different colors
    live_contours = []
    unique_labels = np.unique(separated_cells_mask)
    
    for label in unique_labels:
        if label == 0:  # Skip background
            continue
        
        # Create mask for this specific cell
        cell_mask = (separated_cells_mask == label).astype(np.uint8) * 255
        contours, _ = cv2.findContours(cell_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            live_contours.extend(contours)
    
    # Draw live cell contours in green
    cv2.drawContours(annotated_bgr, live_contours, -1, (0, 255, 0), 2)
    
    # Draw dead cell contours in red
    cv2.drawContours(annotated_bgr, dead_cells, -1, (0, 0, 255), 2)
    
    # Draw watershed boundaries for live cells
    watershed_boundaries = (separated_cells_mask == -1).astype(np.uint8) * 255
    annotated_bgr[watershed_boundaries > 0] = [255, 255, 0]  # Yellow boundaries
    
    # Add text information
    total_count = live_count + dead_count
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(annotated_bgr, f"Live Cells: {live_count}", (10, 30), font, 0.8, (0, 255, 0), 2)
    cv2.putText(annotated_bgr, f"Dead Cells: {dead_count}", (10, 60), font, 0.8, (0, 0, 255), 2)
    cv2.putText(annotated_bgr, f"Total: {total_count}", (10, 90), font, 0.8, (255, 255, 255), 2)
    
    if total_count > 0:
        viability = (live_count / total_count) * 100
        cv2.putText(annotated_bgr, f"Viability: {viability:.1f}%", (10, 120), font, 0.8, (255, 255, 0), 2)
    
    cv2.putText(annotated_bgr, "Method: OTSU + Watershed", (10, 150), font, 0.6, (255, 255, 255), 1)
    
    # Convert back to RGB
    annotated_img = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
    
    return live_count, dead_count, annotated_img, separated_cells_mask, filtered_red_mask

def apply_watershed_segmentation(binary_mask, original_image):
    """
    Apply watershed algorithm to separate overlapping cells using a dual-threshold approach.
    
    Strategy:
    1. Use the provided "loose" binary_mask to identify all potential cell regions
    2. Create a "strict" mask using OTSU to identify only the brightest cell cores as seeds
    3. Use watershed to grow from bright cores within the bounds of the loose mask
    
    Args:
        binary_mask: Binary mask containing all cells (including overlapping ones)
        original_image: Original grayscale image for watershed
    
    Returns:
        cell_count: Number of separated cells
        segmented_mask: Mask with separated cells labeled
    """
    print("      Applying dual-threshold watershed segmentation...")
    
    # Step 1: Keep the existing "loose" binary_mask as the overall cell regions
    
    # Step 2: Create a "strict" mask for "sure foreground" seeds using OTSU
    # This finds only the brightest parts of cells to use as seeds
    _, sure_fg_mask = cv2.threshold(original_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Clean up the strict mask slightly
    kernel_open_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    sure_fg_mask = cv2.morphologyEx(sure_fg_mask, cv2.MORPH_OPEN, kernel_open_small)
    
    # Step 3: Generate seeds from the strict mask (replace peak_local_max approach)
    seed_markers, num_seeds = ndimage.label(sure_fg_mask)
    print(f"      Found {num_seeds} seeds using the Two-Level Threshold method")
    
    # Step 4: Prepare markers for watershed
    # Add 1 so background is 1, not 0
    markers = seed_markers + 1
    
    # Mark the unknown region (inside loose mask but not a seed) as 0
    unknown_region = cv2.subtract(binary_mask, sure_fg_mask)
    markers[unknown_region == 255] = 0
    
    # Prepare the image for watershed
    if len(original_image.shape) == 2:
        original_bgr = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)
    else:
        original_bgr = original_image
    
    # Apply watershed
    markers = cv2.watershed(original_bgr, markers)
    
    # Step 5: Count cells and filter only by size (no solidity filtering)
    unique_labels = np.unique(markers)
    
    # Remove background (label 1) and boundary (label -1)
    cell_labels = [label for label in unique_labels if label > 1]
    
    # Use regionprops for region analysis
    from skimage.measure import regionprops
    
    valid_cells = 0
    final_markers = np.zeros_like(markers)
    min_cell_area = 50  # Restore to a smaller value to include valid small cells
    max_cell_area = 8000
    
    cell_id = 1
    for region in regionprops(markers):
        # Ignore background
        if region.label == 1:
            continue
            
        # Filter only by area - no solidity filtering
        if min_cell_area <= region.area <= max_cell_area:
            valid_cells += 1
            final_markers[markers == region.label] = cell_id
            cell_id += 1
    
    # Mark boundaries
    boundaries = (markers == -1)
    final_markers[boundaries] = -1
    
    print(f"      Watershed found {len(cell_labels)} initial regions, {valid_cells} valid cells after area filtering")
    
    return valid_cells, final_markers

def combine_channels(folder_path, intensity_balance=(1.0, 1.0), brightness_factor=1.0, baseline_percentile=5, save_individual_channels=True):
    """
    Combine green and red channel images and perform advanced RGB-based cell counting.
    """
    default_path = os.path.join(folder_path, "Default")
    
    if not os.path.exists(default_path):
        print(f"Default folder not found in {folder_path}")
        return
    
    # Find green channel image (channel000 - GFP)
    green_files = glob.glob(os.path.join(default_path, "*channel000*.tif"))
    if not green_files:
        print(f"No green channel image found in {default_path}")
        return
    green_file = green_files[0]
    
    # Find red channel image (channel001 - DsRed)
    red_files = glob.glob(os.path.join(default_path, "*channel001*.tif"))
    if not red_files:
        print(f"No red channel image found in {default_path}")
        return
    red_file = red_files[0]
    
    try:
        # Read 16-bit images
        green_img = Image.open(green_file)
        red_img = Image.open(red_file)
        
        print(f"Processing {os.path.basename(folder_path)}:")
        print(f"  Green channel mode: {green_img.mode}, size: {green_img.size}")
        print(f"  Red channel mode: {red_img.mode}, size: {red_img.size}")
        
        # Convert to numpy arrays (preserving 16-bit)
        green_array = np.array(green_img)
        red_array = np.array(red_img)
        
        print(f"  Green stats: min={green_array.min()}, max={green_array.max()}, mean={green_array.mean():.1f}")
        print(f"  Red stats: min={red_array.min()}, max={red_array.max()}, mean={red_array.mean():.1f}")
        
        # Ensure both images have the same size
        if green_array.shape != red_array.shape:
            print(f"Warning: Image shapes don't match in {folder_path}")
            red_array = np.resize(red_array, green_array.shape)
        
        # Apply intensity balance
        green_balanced = green_array * intensity_balance[0]
        red_balanced = red_array * intensity_balance[1]
        
        # Calculate baseline using percentile method for each channel
        green_baseline = np.percentile(green_balanced, baseline_percentile)
        red_baseline = np.percentile(red_balanced, baseline_percentile)
        
        print(f"  Baseline values ({baseline_percentile}th percentile) - Green: {green_baseline:.2f}, Red: {red_baseline:.2f}")
        
        # Subtract baseline from each channel
        green_bg_subtracted = np.maximum(green_balanced - green_baseline, 0)
        red_bg_subtracted = np.maximum(red_balanced - red_baseline, 0)
        
        # Calculate separate scaling factors for each channel after baseline subtraction
        max_green = green_bg_subtracted.max()
        max_red = red_bg_subtracted.max()
        
        # Calculate separate scaling factors for each channel
        if max_green > 0:
            green_scale = (200.0 / max_green) * brightness_factor
        else:
            green_scale = brightness_factor
            
        if max_red > 0:
            red_scale = (200.0 / max_red) * brightness_factor
        else:
            red_scale = brightness_factor
        
        # Convert to 8-bit with channel-specific scaling
        green_8bit = np.clip(green_bg_subtracted * green_scale, 0, 255).astype(np.uint8)
        red_8bit = np.clip(red_bg_subtracted * red_scale, 0, 255).astype(np.uint8)
        
        # Create RGB image for color-based analysis
        height, width = green_8bit.shape
        rgb_array = np.zeros((height, width, 3), dtype=np.uint8)
        rgb_array[:, :, 0] = red_8bit      # Red channel
        rgb_array[:, :, 1] = green_8bit    # Green channel
        rgb_array[:, :, 2] = 0             # Blue channel (empty)
        
        # === ADVANCED RGB-BASED CELL COUNTING ===
        print(f"  Advanced cell counting using OTSU + Watershed...")
        
        live_count, dead_count, cell_analysis_img, green_segmentation, red_mask = count_cells_in_rgb_image(rgb_array)
        
        # Calculate statistics
        total_cells = live_count + dead_count
        if total_cells > 0:
            live_ratio = (live_count / total_cells) * 100
            dead_ratio = (dead_count / total_cells) * 100
        else:
            live_ratio = dead_ratio = 0
        
        print(f"  Advanced Cell Counting Results:")
        print(f"    Live cells (watershed segmented): {live_count}")
        print(f"    Dead cells (red dots): {dead_count}")
        print(f"    Total cells: {total_cells}")
        print(f"    Live cell ratio: {live_ratio:.1f}%")
        print(f"    Dead cell ratio: {dead_ratio:.1f}%")
        
        # Add scale bar to the combined image
        rgb_array_with_scale = add_scale_bar(rgb_array)
        
        # Create PIL image from RGB array with scale bar
        combined_img = Image.fromarray(rgb_array_with_scale, 'RGB')
        
        # Generate output filenames
        base_name = os.path.basename(green_file)
        base_name_clean = base_name.replace("channel000", "combined")
        
        # Save combined image with scale bar
        combined_path = os.path.join(default_path, base_name_clean)
        combined_img.save(combined_path)
        print(f"  ✓ Saved combined image with scale bar: {base_name_clean}")
        
        # Save cell analysis image
        cell_analysis_path = os.path.join(default_path, base_name.replace("channel000", "cell_analysis"))
        Image.fromarray(cell_analysis_img).save(cell_analysis_path)
        print(f"  ✓ Saved cell analysis image: {os.path.basename(cell_analysis_path)}")
        
        # Optionally save individual 8-bit channels
        if save_individual_channels:
            green_solo_path = os.path.join(default_path, base_name.replace("channel000", "green_8bit"))
            red_solo_path = os.path.join(default_path, base_name.replace("channel000", "red_8bit"))
            
            Image.fromarray(green_8bit).save(green_solo_path)
            Image.fromarray(red_8bit).save(red_solo_path)
            print(f"  ✓ Saved 8-bit channels")
        
        # Save additional mask images for debugging
        mask_analysis_path = os.path.join(default_path, base_name.replace("channel000", "color_masks"))
        
        # Create a combined mask visualization
        mask_viz = np.zeros((height, width, 3), dtype=np.uint8)
        # Fix: Use the actual masks returned from the function
        green_mask = (green_segmentation > 0).astype(np.uint8) * 255
        mask_viz[:, :, 1] = green_mask  # Green mask in green channel
        mask_viz[:, :, 0] = red_mask    # Red mask in red channel
        
        Image.fromarray(mask_viz).save(mask_analysis_path)
        print(f"  ✓ Saved color mask analysis: {os.path.basename(mask_analysis_path)}")
        
        # Save additional watershed segmentation visualization
        watershed_viz_path = os.path.join(default_path, base_name.replace("channel000", "watershed_segmentation"))
        
        # Create watershed visualization
        watershed_viz = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Color each segmented cell differently
        unique_labels = np.unique(green_segmentation)
        np.random.seed(42)  # For consistent colors
        
        for label in unique_labels:
            if label <= 0:  # Skip background and boundaries
                continue
            
            # Generate random color for each cell
            color = [np.random.randint(50, 255) for _ in range(3)]
            watershed_viz[green_segmentation == label] = color
        
        # Mark boundaries in white
        watershed_viz[green_segmentation == -1] = [255, 255, 255]
        
        Image.fromarray(watershed_viz).save(watershed_viz_path)
        print(f"  ✓ Saved watershed segmentation: {os.path.basename(watershed_viz_path)}")
        
        # Create enhanced analysis plot with watershed results
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Advanced Cell Analysis (OTSU + Watershed) - {os.path.basename(folder_path)}')
        
        # Original channels
        axes[0,0].imshow(green_8bit, cmap='Greens')
        axes[0,0].set_title(f'Green Channel (Original)')
        axes[0,0].axis('off')
        
        axes[0,1].imshow(red_8bit, cmap='Reds')
        axes[0,1].set_title(f'Red Channel (Original)')
        axes[0,1].axis('off')
        
        # Combined RGB
        axes[0,2].imshow(rgb_array_with_scale)
        axes[0,2].set_title('Combined RGB with Scale Bar')
        axes[0,2].axis('off')
        
        # Watershed segmentation
        axes[1,0].imshow(watershed_viz)
        axes[1,0].set_title(f'Watershed Segmentation ({live_count} cells)')
        axes[1,0].axis('off')
        
        # Final detection results
        axes[1,1].imshow(cell_analysis_img)
        axes[1,1].set_title(f'Final Detection: Live={live_count}, Dead={dead_count}')
        axes[1,1].axis('off')
        
        # Cell statistics pie chart
        if total_cells > 0:
            axes[1,2].pie([live_count, dead_count], 
                         labels=[f'Live ({live_count})', f'Dead ({dead_count})'],
                         colors=['green', 'red'], autopct='%1.1f%%')
            axes[1,2].set_title(f'Cell Viability (Total: {total_cells})')
        else:
            axes[1,2].text(0.5, 0.5, 'No cells detected', ha='center', va='center')
            axes[1,2].set_title('Cell Viability')
        
        plt.tight_layout()
        preview_path = os.path.join(default_path, f"advanced_analysis_{os.path.basename(folder_path)}.png")
        plt.savefig(preview_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved advanced analysis plot: advanced_analysis_{os.path.basename(folder_path)}.png")
        
        # Update results file with advanced method details
        results_path = os.path.join(default_path, f"advanced_cell_results_{os.path.basename(folder_path)}.txt")
        with open(results_path, 'w') as f:
            f.write(f"Advanced Cell Counting Results for {os.path.basename(folder_path)}\n")
            f.write(f"=" * 60 + "\n")
            f.write(f"Advanced Detection Methods:\n")
            f.write(f"- Green channel: OTSU automatic thresholding + Watershed segmentation\n")
            f.write(f"- Red channel: OTSU thresholding + area filtering\n")
            f.write(f"- Improvements: Better detection of dim cells, separation of overlapping cells\n")
            f.write(f"\n")
            f.write(f"Technical Details:\n")
            f.write(f"- OTSU thresholding: Automatic optimal threshold calculation\n")
            f.write(f"- Watershed segmentation: Distance transform + connected components\n")
            f.write(f"- Cell area filtering: 100-8000 pixels for live cells, 5-200 pixels for dead cells\n")
            f.write(f"\n")
            f.write(f"Results:\n")
            f.write(f"Live cells (watershed segmented): {live_count}\n")
            f.write(f"Dead cells (OTSU + area filtered): {dead_count}\n")
            f.write(f"Total cells: {total_cells}\n")
            f.write(f"Live cell ratio: {live_ratio:.2f}%\n")
            f.write(f"Dead cell ratio: {dead_ratio:.2f}%\n")
        print(f"  ✓ Saved advanced results: advanced_cell_results_{os.path.basename(folder_path)}.txt")
        
        return {
            'folder': os.path.basename(folder_path),
            'live_cells': live_count,
            'dead_cells': dead_count,
            'total_cells': total_cells,
            'live_ratio': live_ratio,
            'dead_ratio': dead_ratio
        }
        
    except Exception as e:
        print(f"Error processing images in {folder_path}: {str(e)}")
        return None

def main():
    """
    Main function to process all folders with cell counting.
    """
    base_path = '/Users/kiki/Desktop/report/ test rohit'
    
    # List of folder names to process
    folders_to_process = ["A4.1_1", "A4.2_1", "a5.1_1"]
    
    # Find additional folders automatically
    if os.path.exists(base_path):
        all_folders = [f for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f))]
        for folder in all_folders:
            if folder not in folders_to_process and any(char.isdigit() for char in folder):
                folders_to_process.append(folder)
    
    print(f"Processing folders: {folders_to_process}")
    print("="*50)
    
    # Parameters you can adjust:
    intensity_balance = (1.0, 1.0)  # (green_multiplier, red_multiplier)
    brightness_factor = 1.5  # Brightness control
    baseline_percentile = 10  # Remove bottom 10% as baseline
    
    # Store results for summary
    all_results = []
    
    for folder_name in folders_to_process:
        folder_path = os.path.join(base_path, folder_name)
        if os.path.exists(folder_path):
            result = combine_channels(folder_path, 
                                    intensity_balance=intensity_balance, 
                                    brightness_factor=brightness_factor, 
                                    baseline_percentile=baseline_percentile)
            if result:
                all_results.append(result)
        else:
            print(f"Folder not found: {folder_path}")
        print("-"*30)
    
    # Print summary of all results
    if all_results:
        print("\n" + "="*60)
        print("CELL COUNTING SUMMARY")
        print("="*60)
        total_live = sum(r['live_cells'] for r in all_results)
        total_dead = sum(r['dead_cells'] for r in all_results)
        total_all = total_live + total_dead
        
        for result in all_results:
            print(f"{result['folder']}: Live={result['live_cells']}, Dead={result['dead_cells']}, "
                  f"Total={result['total_cells']}, Viability={result['live_ratio']:.1f}%")
        
        print("-"*60)
        if total_all > 0:
            overall_viability = (total_live / total_all) * 100
            print(f"OVERALL: Live={total_live}, Dead={total_dead}, "
                  f"Total={total_all}, Viability={overall_viability:.1f}%")
        print("="*60)
        
        # Generate summary table file in main directory
        try:
            # Create CSV file for easy import into Excel
            csv_path = os.path.join(base_path, "cell_count_summary.csv")
            with open(csv_path, 'w') as f:
                # Write header
                f.write("Sample,Live Cells,Dead Cells,Total Cells,Viability (%)\n")
                
                # Write data for each sample
                for result in all_results:
                    f.write(f"{result['folder']},{result['live_cells']},{result['dead_cells']},"
                            f"{result['total_cells']},{result['live_ratio']:.2f}\n")
                
                # Write overall totals
                if total_all > 0:
                    f.write(f"OVERALL,{total_live},{total_dead},{total_all},{overall_viability:.2f}\n")
            
            print(f"\nSummary table saved to: {csv_path}")
            
            # Also create a more readable text table
            txt_path = os.path.join(base_path, "cell_count_summary.txt")
            with open(txt_path, 'w') as f:
                f.write("Cell Count Summary Table\n")
                f.write("=" * 60 + "\n\n")
                
                # Format header with fixed width columns
                f.write(f"{'Sample':<15}{'Live Cells':<12}{'Dead Cells':<12}{'Total Cells':<12}{'Viability (%)':<15}\n")
                f.write("-" * 60 + "\n")
                
                # Write data rows
                for result in all_results:
                    f.write(f"{result['folder']:<15}{result['live_cells']:<12}{result['dead_cells']:<12}"
                            f"{result['total_cells']:<12}{result['live_ratio']:.2f}%\n")
                
                # Write totals
                if total_all > 0:
                    f.write("-" * 60 + "\n")
                    f.write(f"{'OVERALL':<15}{total_live:<12}{total_dead:<12}{total_all:<12}{overall_viability:.2f}%\n")
            
            print(f"Formatted text table saved to: {txt_path}")
            
        except Exception as e:
            print(f"Error creating summary table: {str(e)}")
    
    print("\nProcessing complete!")
    print("New outputs:")
    print("- Cell analysis images showing detected cell boundaries")
    print("- Enhanced analysis plots with cell counts and viability pie charts")
    print("- Text files with detailed cell counting results")
    print("- Summary tables (CSV and TXT) in main directory")

if __name__ == "__main__":
    main()
