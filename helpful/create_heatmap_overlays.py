import fast
import os
fast.Reporter.setGlobalReportMethod(fast.Reporter.COUT)
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Cursor
import tifffile
from pathlib import Path


model = 'clam_mb121224_205925_s1'
size = '1024'

root_dir = Path(f'/mnt/EncryptedDisk2/BreastData/Studies/CLAM/heatmaps/{size}/{model}/heatmap_production_results/HEATMAP_OUTPUT/Unspecified/')
save_dir = Path(f'/mnt/EncryptedDisk2/BreastData/Studies/CLAM/heatmaps/{size}/{model}/heatmap_overlay')

if not os.path.exists(save_dir):
    os.makedirs(save_dir)
image_list = []
for image in Path(root_dir).iterdir():
    slide_id = image.stem.split('_')[0]
    # Append to image_list if it's not already in it
    if slide_id not in image_list:
        image_list.append(slide_id)

for slide_id in image_list :
    if (save_dir / f'{slide_id}.jpg').exists():
        print(f'{slide_id}.jpg does already exists.')
        continue
    attention_path = f'/mnt/EncryptedDisk2/BreastData/Studies/CLAM/heatmaps/{size}/{model}/heatmap_production_results/HEATMAP_OUTPUT/Unspecified/{slide_id}_0.5_roi_0_blur_0_rs_0_bc_0_a_0.4_l_-1_bi_0_-1.0.jpg'
    actual_path = f'/mnt/EncryptedDisk2/BreastData/Studies/CLAM/heatmaps/{size}/{model}/heatmap_production_results/HEATMAP_OUTPUT/Unspecified/{slide_id}_orig_0.jpg'

    try:
        attention_image_importer = fast.ImageImporter.create(attention_path)
        image_importer = fast.ImageImporter.create(actual_path)

        #
        a_image = attention_image_importer.runAndGetOutputData()
        orig_image = image_importer.runAndGetOutputData()

        n = np.asarray(a_image)
        orig = np.asarray(orig_image)

        fig, axs = plt.subplots(3, 1, dpi=400)

        # Display the first image in the first subplot
        axs[0].imshow(a_image, cmap='gray')
        axs[0].set_title('Attention map')
        axs[0].axis('off')  # Hide axes for a cleaner look

        # Display the second image in the second subplot
        axs[1].imshow(orig, cmap='gray')
        axs[1].set_title('Original Image')
        axs[1].axis('off')  # Hide axes for a cleaner look

        axs[2].imshow(orig, cmap='gray')
        axs[2].imshow(a_image, cmap='jet', alpha=0.25)  # Use 'alpha' for transparency
        axs[2].set_title('Overlay')
        axs[2].axis('off')

        # Adjust the layout so the subplots fit well within the figure area
        plt.tight_layout()

        # Show the plot
        plt.savefig(save_dir / f'{slide_id}.jpg')
        print(f'Save {slide_id}.jpg')
        plt.close()
    except:
        print(f'{slide_id} could not be opened')

