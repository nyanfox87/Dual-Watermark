import os

import numpy as np
from PIL import Image

from adapters import EditGuardAdapter, StegaStampAdapter
from services.pipeline import run_embed_pipeline
from services.verify import run_verify_pipeline

image = np.array(Image.open('/home/project/Documents/ttttt.png').convert('RGB'), dtype=np.uint8)
edit_bits = '0101010101010101010101010101010101010101010101010101010101010101'
secret = 'Stega!'

editguard = EditGuardAdapter('/home/project/Documents/EditGuard')
stega = StegaStampAdapter('/home/project/Documents/StegaStamp-pytorch')

embed = run_embed_pipeline(
    image_rgb_uint8=image,
    editguard_bits=edit_bits,
    stegastamp_secret=secret,
    editguard_adapter=editguard,
    stegastamp_adapter=stega,
    stegastamp_model_dir='/home/project/Documents/StegaStamp-pytorch/asset/best.pth',
)

stega_img = embed['stegastamp_image']
final_img = embed['final_image']
os.makedirs('./output', exist_ok=True)
Image.fromarray(stega_img).save('./output/stegastamp_stage.png')
Image.fromarray(final_img).save('./output/final_watermarked.png')

verify = run_verify_pipeline(
    image_rgb_uint8=final_img,
    metadata_json=embed['metadata_json'],
    editguard_adapter=editguard,
    stegastamp_adapter=stega,
    detector_model_dir='',
    stegastamp_model_dir='/home/project/Documents/StegaStamp-pytorch/asset/best.pth',
)

print('decoded_texts=', verify['stegastamp_found_codes'])
print('editguard_accuracy=', verify['editguard_accuracy'])
print('editguard_bits_len=', len(verify['editguard_recovered_bits']))
print('mask_shape=', verify['editguard_mask'].shape)
print('overall_pass=', verify['summary']['overall_pass'])
print('has_stega_stage=', 'stegastamp_image' in embed)
print('same_stega_and_final=', np.array_equal(stega_img, final_img))
